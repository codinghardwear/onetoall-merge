#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <omp.h>
#include <chrono>
#include <thread>
#include <algorithm>

#include "../hnsw/hnswlib/hnswlib.h"
#include "util/timer.h"
#include "util/multi_thread.h"
#include "util/graph.h"

int c = 0;

// Construction parameters
int dim = 128;   int max_elements = 1000000;
int M = 16;     int ef_construction = 200;      int ef_search = 10;

hnswlib::L2SpaceI space(dim);

Timer timer;

int main() {

    std::string sub_hnsw1_path = "sub1_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw1 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw1_path);        
    cout << "Sub hnsw1 loaded \t";
    std::string sub_hnsw2_path = "sub2_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw2 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw2_path);        
    cout << "Sub hnsw2 loaded \t";
    std::string sub_hnsw3_path = "sub3_hnsw_1M_t80_f.bin";
    hnswlib::HierarchicalNSW<int>* sub_hnsw3 = new hnswlib::HierarchicalNSW<int>(&space, sub_hnsw3_path);        
    cout << "Sub hnsw3 loaded" << endl;

    int hnsw1_elements = sub_hnsw1->cur_element_count;      
    int hnsw2_elements = sub_hnsw2->cur_element_count;
    int hnsw3_elements = sub_hnsw3->cur_element_count;

    cout << "sub graph elements: " << hnsw1_elements << " " << hnsw2_elements << " " << hnsw3_elements << endl;
    cout << endl;

    // timer.reset();
    // DistanceGraph graph_U0(max_elements, 2 * M);
    // #pragma omp parallel
    // {
    //     #pragma omp for
    //     for (int id = 0; id < hnsw1_elements; id++) {
    //         hnswlib::linklistsizeint* data = sub_hnsw1->get_linklist0(id);
    //         int size = sub_hnsw1->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = sub_hnsw1->fstdistfunc_(sub_hnsw1->getDataByInternalId(id), sub_hnsw1->getDataByInternalId(neighbor_id), sub_hnsw1->dist_func_param_);
    //             graph_U0.addNeighbor(id, neighbor_id, dist);
    //         }
    //     }
    //     #pragma omp for
    //     for (int id = 0; id < hnsw2_elements; id++) {
    //         hnswlib::linklistsizeint* data = sub_hnsw2->get_linklist0(id);
    //         int size = sub_hnsw2->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = sub_hnsw2->fstdistfunc_(sub_hnsw2->getDataByInternalId(id), sub_hnsw2->getDataByInternalId(neighbor_id), sub_hnsw2->dist_func_param_);
    //             graph_U0.addNeighbor(id + hnsw1_elements, neighbor_id, dist);
    //         }
    //     }   
    //     #pragma omp for
    //     for (int id = 0; id < hnsw3_elements; id++) {
    //         hnswlib::linklistsizeint* data = sub_hnsw3->get_linklist0(id);
    //         int size = sub_hnsw3->getListCount(data);
    //         hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //         for (int j = 0; j < size; j++) {
    //             int neighbor_id = datal[j];
    //             float dist = sub_hnsw3->fstdistfunc_(sub_hnsw3->getDataByInternalId(id), sub_hnsw3->getDataByInternalId(neighbor_id), sub_hnsw3->dist_func_param_);
    //             graph_U0.addNeighbor(id + (hnsw1_elements + hnsw2_elements), neighbor_id, dist);
    //         }
    //     }   
    // }
    // cout << "Level 0 graph_U0 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // graph_U0.printGraph(hnsw1_elements - 3, hnsw1_elements);
    // graph_U0.printGraph((hnsw1_elements + hnsw2_elements) - 3, (hnsw1_elements + hnsw2_elements));
    // graph_U0.printGraph((hnsw1_elements + hnsw2_elements + hnsw3_elements) - 3, (hnsw1_elements + hnsw2_elements + hnsw3_elements));

    std::vector<int> level1_nodes;
    for (int id = 0; id < hnsw3_elements; id++) {
        if (sub_hnsw3->element_levels_[id] != 0) {
            level1_nodes.push_back(id);
        }
    }

    cout << "level 1 has " << level1_nodes.size() << " nodes" << endl;

    timer.reset();
    DistanceGraph graph_U1(hnsw3_elements, M);
    // #pragma omp parallel for
    for (int id : level1_nodes) {
        if (sub_hnsw3->linkLists_[id] == nullptr) {
            cout << id << "does not allocate memory" << endl;
        }
        else {
            hnswlib::linklistsizeint* data = sub_hnsw3->get_linklist(id, 1);
            if (data == nullptr) {
                cout << id << endl;
                return 0;
            }
            int size = sub_hnsw3->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                int neighbor_id = datal[j];
                float dist = sub_hnsw3->fstdistfunc_(sub_hnsw3->getDataByInternalId(id), sub_hnsw3->getDataByInternalId(neighbor_id), sub_hnsw3->dist_func_param_);
                graph_U1.addNeighbor(id, neighbor_id, dist);
            }
        }
    }

    graph_U1.printGraph(level1_nodes[0], level1_nodes[0]);
    graph_U1.printGraph(level1_nodes[level1_nodes.size()-1], level1_nodes[level1_nodes.size()-1]);

    return 0;
}