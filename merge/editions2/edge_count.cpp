#include <omp.h>
#include <chrono>
#include <thread>
#include <algorithm>

#include "../hnsw/hnswlib/hnswlib.h"
#include "./util/timer.h"
#include "./util/multi_thread.h"
#include "./util/graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int max_elements = 1000000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2Space space(dim);

Timer timer;

int main() {

    int hnsw1_elements = max_elements / 2;    
    int hnsw2_elements = max_elements / 2;
    int cur_elements = hnsw1_elements + hnsw2_elements;

    // std::uniform_real_distribution<> distrib_real;
    // std::mt19937 rng;  rng.seed(0);
    // float* data = new float[dim * cur_elements];
    // for (int i = 0; i < dim * cur_elements; i++) {
    //     data[i] = distrib_real(rng);  
    // }

    std::string hnsw_path = "mer_hnsw_10M_t80.bin";
    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    merged_hnsw->ef_ = ef_search;        
    cout << "Merged hnsw loaded" << endl;

    // timer.reset();
    // int correct = 0;
    // std::vector<hnswlib::labeltype> record(cur_elements);
    // #pragma omp parallel for
    // for (int i = 0; i < cur_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     record[i] = label;
    // }
    // for (int i = 0; i < cur_elements; i++) {
    //     if (record[i] == i)
    //         correct++;
    // }
    // cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // float recall = (float)correct / cur_elements;
    // std::cout << "Recall of Merged graph: " << recall << "\n";

    int m_edge_count = 0;
    DistanceGraph graph_U0m(cur_elements, 2 * M);
    for (int id = 0; id < cur_elements; id++) {
        hnswlib::linklistsizeint* data = merged_hnsw->get_linklist0(id);
        int size = merged_hnsw->getListCount(data);
        m_edge_count += size;
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int j = 0; j < size; j++) {
            int neighbor_id = datal[j];
            float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(neighbor_id), merged_hnsw->dist_func_param_);
            graph_U0m.addNeighbor(id, neighbor_id, -dist);
        }
    }

    int cross_edge_count = 0;
    for (int id = 0; id < hnsw1_elements; id++) {
        auto pq = graph_U0m.getNeighbors(id);
        while (!pq.empty()) {
            if (pq.top().first >= hnsw1_elements)
                cross_edge_count++;
            pq.pop();
        }
    }
    for (int id = hnsw1_elements; id < cur_elements; id++) {
        auto pq = graph_U0m.getNeighbors(id);
        while (!pq.empty()) {
            if (pq.top().first < hnsw1_elements)
                cross_edge_count++;
            pq.pop();
        }
    }
    cout << "Merged hnsw has " << m_edge_count << " edges" << endl;
    cout << "Merged hnsw has " << cross_edge_count << " cross edges" << endl;
    cout << "The ratio of cross edges / total edges is " << (float)cross_edge_count / m_edge_count << endl;
    cout << endl;


    hnsw_path = "complete_hnsw_100k.bin";
    hnswlib::HierarchicalNSW<float>* complete_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);        
    cout << "Complete hnsw loaded" << endl;

    int c_edge_count = 0;
    DistanceGraph graph_U0c(cur_elements, 2 * M);
    for (int id = 0; id < cur_elements; id++) {
        hnswlib::linklistsizeint* data = complete_hnsw->get_linklist0(id);
        int size = complete_hnsw->getListCount(data);
        c_edge_count += size;
        hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
        for (int j = 0; j < size; j++) {
            int neighbor_id = datal[j];
            float dist = complete_hnsw->fstdistfunc_(complete_hnsw->getDataByInternalId(id), complete_hnsw->getDataByInternalId(neighbor_id), complete_hnsw->dist_func_param_);
            graph_U0c.addNeighbor(id, neighbor_id, -dist);
        }
    }
    cout << "Complete hnsw has " << c_edge_count << " edges" << endl;

    float dist = 0;
    for (int id = 0; id < cur_elements; id++) {
        auto pqm = graph_U0m.getNeighbors(id);
        auto pqc = graph_U0c.getNeighbors(id);

        while (!pqc.empty() && !pqm.empty()) {
            dist += -pqc.top().second + pqm.top().second;
            pqc.pop();
            pqm.pop();
        }
    }
    cout << "Total distance improved is " << dist << " and distance improved per edge is " << dist / c_edge_count << endl;

    cross_edge_count = 0;
    for (int id = 0; id < hnsw1_elements; id++) {
        auto pq = graph_U0c.getNeighbors(id);
        while (!pq.empty()) {
            if (pq.top().first >= hnsw1_elements)
                cross_edge_count++;
            pq.pop();
        }
    }
    for (int id = hnsw1_elements; id < cur_elements; id++) {
        auto pq = graph_U0c.getNeighbors(id);
        while (!pq.empty()) {
            if (pq.top().first < hnsw1_elements)
                cross_edge_count++;
            pq.pop();
        }
    }
    cout << "Complete hnsw has " << c_edge_count << " edges" << endl;
    cout << "Complete hnsw has " << cross_edge_count << " cross edges" << endl;
    cout << "The ratio of cross edges / total edges is " << (float)cross_edge_count / c_edge_count << endl;

    return 0;
}