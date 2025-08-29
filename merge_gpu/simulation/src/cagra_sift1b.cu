/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

/**
 * @brief Reads vectors from a file in fvecs format.
 *
 * The fvecs format consists of a 4-byte integer (vector dimension),
 * followed by vectors stored as 4-byte floats.
 *
 * @param handle RAFT device resources handle
 * @param file_path Path to the fvecs file
 * @param dim Output parameter for vector dimension
 * @param n_rows Output parameter for the number of vectors read
 * @param max_rows_to_read The maximum number of vectors to read from the file. -1 means all.
 * @return raft::device_matrix<float, int64_t> A device matrix containing the data
 */
raft::device_matrix<float, int64_t> read_fvecs(raft::device_resources const& handle,
                                               const std::string& file_path,
                                               int64_t& dim,
                                               int64_t& n_rows,
                                               int64_t max_rows_to_read = -1)
{
  std::ifstream is(file_path, std::ios::binary);
  if (!is.is_open()) { throw std::runtime_error("Could not open file: " + file_path); }

  // Read dimension
  int d;
  is.read(reinterpret_cast<char*>(&d), sizeof(int));
  dim = static_cast<int64_t>(d);

  // Get file size to calculate total number of vectors in file
  is.seekg(0, std::ios::end);
  size_t file_size = is.tellg();
  is.seekg(0, std::ios::beg);
  size_t vector_size_bytes = (1 * sizeof(int)) + (dim * sizeof(float));
  int64_t total_rows_in_file = file_size / vector_size_bytes;

  // Determine how many rows to read
  n_rows = total_rows_in_file;
  if (max_rows_to_read > 0 && max_rows_to_read < total_rows_in_file) {
    n_rows = max_rows_to_read;
  }

  // Read data into host memory
  std::cout << "Loading " << n_rows << " vectors of dimension " << dim << " from " << file_path
            << std::endl;
  std::vector<float> host_data(n_rows * dim);
  for (int64_t i = 0; i < n_rows; ++i) {
    is.seekg(sizeof(int), std::ios::cur);  // Skip dimension
    is.read(reinterpret_cast<char*>(host_data.data() + i * dim), dim * sizeof(float));
  }
  is.close();

  // Copy to device
  auto device_mat = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  raft::update_device(device_mat.data_handle(), host_data.data(), n_rows * dim, handle.get_stream());
  handle.sync_stream();
  return device_mat;
}

/**
 * @brief Reads vectors from a file in ivecs format.
 *
 * The ivecs format is similar to fvecs but with integer data.
 *
 * @param handle RAFT device resources handle
 * @param file_path Path to the ivecs file
 * @param dim Output parameter for vector dimension
 * @param n_rows Output parameter for the number of vectors
 * @return raft::device_matrix<uint32_t, int64_t> A device matrix containing the data
 */
raft::device_matrix<uint32_t, int64_t> read_ivecs(raft::device_resources const& handle,
                                                  const std::string& file_path,
                                                  int64_t& dim,
                                                  int64_t& n_rows)
{
  std::ifstream is(file_path, std::ios::binary);
  if (!is.is_open()) { throw std::runtime_error("Could not open file: " + file_path); }

  // Read dimension
  int d;
  is.read(reinterpret_cast<char*>(&d), sizeof(int));
  dim = static_cast<int64_t>(d);

  // Get file size to calculate number of vectors
  is.seekg(0, std::ios::end);
  size_t file_size = is.tellg();
  is.seekg(0, std::ios::beg);
  size_t vector_size_bytes = (1 * sizeof(int)) + (dim * sizeof(int));
  n_rows                 = file_size / vector_size_bytes;

  // Read data into host memory
  std::cout << "Loading " << n_rows << " vectors of dimension " << dim << " from " << file_path
            << std::endl;
  std::vector<uint32_t> host_data(n_rows * dim);
  for (int64_t i = 0; i < n_rows; ++i) {
    is.seekg(sizeof(int), std::ios::cur);  // Skip dimension
    is.read(reinterpret_cast<char*>(host_data.data() + i * dim), dim * sizeof(uint32_t));
  }
  is.close();

  // Copy to device
  auto device_mat = raft::make_device_matrix<uint32_t, int64_t>(handle, n_rows, dim);
  raft::update_device(device_mat.data_handle(), host_data.data(), n_rows * dim, handle.get_stream());
  handle.sync_stream();
  return device_mat;
}

/**
 * @brief Calculates the recall@K metric.
 *
 * Recall is the fraction of true nearest neighbors that are found by the search.
 *
 * @param handle RAFT device resources handle
 * @param neighbors The neighbor indices found by the search algorithm
 * @param ground_truth The ground truth neighbor indices
 */
void calculate_recall(raft::device_resources const& handle,
                      raft::device_matrix_view<const uint32_t, int64_t> neighbors,
                      raft::device_matrix_view<const uint32_t, int64_t> ground_truth)
{
  handle.sync_stream();
  int64_t n_queries = neighbors.extent(0);
  int64_t topk      = neighbors.extent(1);
  int64_t gt_k      = ground_truth.extent(1);

  // Copy data to host for calculation
  auto host_neighbors    = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  auto host_ground_truth = raft::make_host_matrix<uint32_t, int64_t>(n_queries, gt_k);
  raft::update_host(
    host_neighbors.data_handle(), neighbors.data_handle(), n_queries * topk, handle.get_stream());
  raft::update_host(host_ground_truth.data_handle(),
                    ground_truth.data_handle(),
                    n_queries * gt_k,
                    handle.get_stream());
  handle.sync_stream();

  int total_found = 0;
  for (int64_t i = 0; i < n_queries; ++i) {
    std::unordered_set<uint32_t> gt_set;
    for (int64_t j = 0; j < topk; ++j) {
      gt_set.insert(host_ground_truth(i, j));
    }

    for (int64_t j = 0; j < topk; ++j) {
      if (gt_set.count(host_neighbors(i, j))) { total_found++; }
    }
  }

  float recall = static_cast<float>(total_found) / (static_cast<float>(n_queries) * topk);
  std::cout << "Recall@" << topk << ": " << recall << std::endl;
}

/**
 * @brief Build CAGRA index, search, and evaluate on SIFT1B dataset.
 *
 * @param dev_resources RAFT device resources handle
 * @param dataset The training dataset (e.g., sift1b_base)
 * @param queries The query vectors (e.g., sift1b_query)
 * @param ground_truth The ground truth nearest neighbors for the queries
 */
void cagra_on_sift1b(raft::device_resources const& dev_resources,
                     raft::device_matrix_view<const float, int64_t> dataset,
                     raft::device_matrix_view<const float, int64_t> queries,
                     raft::device_matrix_view<const uint32_t, int64_t> ground_truth)
{
  using namespace cuvs::neighbors;

  // We will search for the top-K neighbors.
  // The ground truth file might contain more neighbors than we want to search for (e.g., top-100),
  // so we'll use a smaller `topk` for the actual search and recall calculation.
  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);

  // Create output arrays on the device
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Configure CAGRA index parameters
  cagra::index_params index_params;
  index_params.intermediate_graph_degree = 128;
  index_params.graph_degree              = 64;

  std::cout << "Building CAGRA index (search graph)..." << std::endl;
  auto start_build = std::chrono::high_resolution_clock::now();
  auto index       = cagra::build(dev_resources, index_params, dataset);
  dev_resources.sync_stream();  // Ensure build is complete before stopping timer
  auto stop_build = std::chrono::high_resolution_clock::now();
  auto build_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(stop_build - start_build);
  std::cout << "Index construction time: " << build_duration.count() << " ms" << std::endl;

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // Configure search parameters
  cagra::search_params search_params;
  search_params.max_queries = 1000;  // Batch size for search

  // Search K nearest neighbors
  std::cout << "Searching for " << topk << " nearest neighbors..." << std::endl;
  auto start_search = std::chrono::high_resolution_clock::now();
  cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  dev_resources.sync_stream();  // Ensure search is complete before stopping timer
  auto stop_search = std::chrono::high_resolution_clock::now();
  auto search_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(stop_search - start_search);

  std::cout << "Search time: " << search_duration.count() << " ms" << std::endl;
  std::cout << "Search throughput: "
            << static_cast<double>(n_queries) / (search_duration.count() / 1000.0) << " QPS"
            << std::endl;

  // Evaluate the results
  std::cout << "Calculating recall..." << std::endl;
  
  // **FIXED LINE:** Create a sub-view of the ground truth to match the number of neighbors we searched for (topk).
  // This is the correct way to create a sub-view for this RAFT version.
  auto ground_truth_view = raft::make_device_matrix_view(
    ground_truth.data_handle(), ground_truth.extent(0), topk);
    
  calculate_recall(dev_resources, neighbors.view(), ground_truth_view);
}

int main(int argc, char** argv)
{
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " <base_fvecs_path> <query_fvecs_path> <groundtruth_ivecs_path> "
                 "[num_dataset_vectors]"
              << std::endl;
    std::cerr << "  [num_dataset_vectors] (optional): Number of vectors to use from the base file. "
                 "If not specified, all vectors are used."
              << std::endl;
    std::cerr << "Example (1M vectors): " << argv[0]
              << " sift1b_base.fvecs sift1b_query.fvecs sift1b_groundtruth.ivecs 1000000"
              << std::endl;
    return 1;
  }

  std::string base_path                  = argv[1];
  std::string query_path                 = argv[2];
  std::string gt_path                    = argv[3];
  long long num_dataset_vectors_to_use = -1;
  if (argc == 5) { num_dataset_vectors_to_use = std::stoll(argv[4]); }

  raft::device_resources dev_resources;

  // With ample system RAM, we can remove the RMM pool memory resource.
  // This allows RMM to use the default cudaMalloc/cudaFree for memory management,
  // which can be simpler and avoid pool-size limitations on high-memory systems.
  //
  // rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
  //   rmm::mr::get_current_device_resource(), 4ull * 1024 * 1024 * 1024);
  // rmm::mr::set_current_device_resource(&pool_mr);

  // Load datasets from files
  int64_t n_samples, n_dim_samples, n_queries, n_dim_queries, n_gt, n_dim_gt;
  auto dataset =
    read_fvecs(dev_resources, base_path, n_dim_samples, n_samples, num_dataset_vectors_to_use);
  auto queries      = read_fvecs(dev_resources, query_path, n_dim_queries, n_queries);
  auto ground_truth = read_ivecs(dev_resources, gt_path, n_dim_gt, n_gt);

  if (n_dim_samples != n_dim_queries) {
    std::cerr << "Error: Dataset and query dimensions do not match (" << n_dim_samples << " vs "
              << n_dim_queries << ")" << std::endl;
    return 1;
  }
  if (n_queries != n_gt) {
    std::cerr << "Error: Number of queries and ground truth entries do not match (" << n_queries
              << " vs " << n_gt << ")" << std::endl;
    return 1;
  }

  // Run the build, search, and evaluation
  cagra_on_sift1b(dev_resources,
                  raft::make_const_mdspan(dataset.view()),
                  raft::make_const_mdspan(queries.view()),
                  raft::make_const_mdspan(ground_truth.view()));

  return 0;
}
