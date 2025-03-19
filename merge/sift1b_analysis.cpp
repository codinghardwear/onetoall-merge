#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>


std::vector<unsigned char> load_bvecs(const std::string& path, size_t num_vectors, size_t dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::vector<unsigned char> data(num_vectors * dim);
    for (size_t i = 0; i < num_vectors; i++) {
        int d;
        input.read((char *)(&d), sizeof(int));
        if (d != dim) {
            std::cerr << "Expected dimension: " << dim << ", found dimension: " << d << " at vector " << i << std::endl;
            throw std::runtime_error("Dimension mismatch in bvecs file.");
        }
        input.read((char *)(data.data() + i * dim), dim);
    }

    return data;
}

void save_histograms(const std::vector<std::vector<int>>& histograms, const std::string& output_path) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Cannot open output file");
    }

    for (size_t d = 0; d < histograms.size(); ++d) {
        out << "dim" << d << ((d != histograms.size()-1) ? "," : "\n");
    }

    for (int val = 0; val < 256; ++val) {
        for (size_t d = 0; d < histograms.size(); ++d) {
            out << histograms[d][val] 
               << ((d != histograms.size()-1) ? "," : "\n");
        }
    }
}

int main() {
    const size_t dim = 128;
    const size_t analyze_num = 1000000000;

    try {
        std::string base_path = "/media/raid5/myt/bigann_base.bvecs";
        std::cout << "Loading first " << analyze_num << " vectors..." << std::endl;
        auto base_vectors = load_bvecs(base_path, analyze_num, dim);

        std::vector<std::vector<int>> histograms(dim, std::vector<int>(256, 0));

        std::cout << "Analyzing distributions..." << std::endl;
        // omp_set_num_threads(80);
        // #pragma omp parallel for
        for (size_t i = 0; i < analyze_num; ++i) {
            const auto* vec = &base_vectors[i * dim];
            for (size_t d = 0; d < dim; ++d) {
                histograms[d][vec[d]]++;
            }
        }

        std::string output_path = "dim_distributions.csv";
        save_histograms(histograms, output_path);
        std::cout << "Analysis saved to: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}