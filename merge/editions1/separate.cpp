#include "../hnsw/hnswlib/hnswlib.h"

#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>

#include "timer.h"
#include "multi_thread.h"
#include "graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters
int dim = 128;   int maxElementsNum = 50000;
int M = 16;     int ef_construction = 200;      int ef_search = 15;

Timer timer;

hnswlib::L2Space space(dim);

struct CompareBySecond {
    constexpr bool operator()(std::pair<int, float> const& a, std::pair<int, float> const& b) const noexcept {
        return a.second < b.second;
    }
};

void getNeighborsByHeuristic(std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, CompareBySecond> &top_candidates, const size_t M) {
    
    if (top_candidates.size() < M) {
        return;
    }

    std::priority_queue<std::pair<int, float>> queue_closest;
    std::vector<std::pair<int, float>> return_list;
    
    while (top_candidates.size() > 0) {
        queue_closest.emplace(top_candidates.top().first, -top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M) {
            break;
        }
        std::pair<int, float> curent_pair = queue_closest.top();
        float dist_to_query = -curent_pair.second;
        queue_closest.pop();
        bool good = true;

        for (std::pair<int, float> second_pair : return_list) {
            float curdist = fstdistfunc_(getDataByInternalId(second_pair.first),getDataByInternalId(curent_pair.first), dist_func_param_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
                return_list.push_back(curent_pair);
        }
    }

    for (std::pair<int, float> curent_pair : return_list) {
        top_candidates.emplace(current_pair.first, -curent_pair.second);
    }
}

int main() {

    int graph1ElementsNum = maxElementsNum / 2;    int graph2ElementsNum = maxElementsNum / 2;
    int curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, graph1ElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, graph2ElementsNum, M, ef_construction);

    std::string hnsw_path1 = "sub_hnsw1_25k_t1.bin";
    std::string hnsw_path2 = "sub_hnsw2_25k_t1.bin";

    hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);        cout << "graph1" << endl;
    hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);        cout << "graph2" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, curElementsNum, M, ef_construction);

    std::string hnsw_path = "merged_hnsw_50k.bin";

    merged_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);       cout << "graph" << endl;
    cout << endl;

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * curElementsNum];
    for (size_t i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }

    // cout << merged_hnsw->element_levels_[5545] << endl;

    // Level 1 to Level max
    std::vector<std::vector<int>> nodes_level(merged_hnsw->maxlevel_ + 1);
    for (int id = 0; id < curElementsNum; id++) {
        if (merged_hnsw->linkLists_[id] != nullptr) {
            int level = merged_hnsw->element_levels_[id];
            for (int i = level; i >= 1; i--) {
                nodes_level[i].push_back(id);
            }
        }
    }
    // for (int id : nodes_level[1]) {
    //     cout << id << " ";
    // }
    // cout << endl;

    // std::vector<std::vector<int>> nodes_level2(merged_hnsw->maxlevel_ + 1);
    // for (int id = 0; id < curElementsNum; id++) {
    //     if (merged_hnsw->linkLists_[id] != nullptr) {
    //         int level = merged_hnsw->element_levels_[id];
    //         nodes_level2[level].push_back(id);
    //     }
    // }
    // for (int id : nodes_level2[1]) {
    //     cout << id << " ";
    // }
    // cout << endl;

    // for (int i = 0; i < nodes_level2.size(); i++) {
    //     if (nodes_level[1][i] != nodes_level2[1][i]) {
    //         cout << "NO" << endl;
    //     }
    // }


    // auto find_id = std::find(nodes_level[1].begin(), nodes_level[1].end(), 17201);
    // if (find_id != nodes_level[1].end()) {
    //     cout << "Find" << endl;
    // }
    // find_id = std::find(nodes_level[1].begin(), nodes_level[1].end(), 5545);
    // if (find_id != nodes_level[1].end()) {
    //     cout << "Find" << endl;
    // }

    // cout << nodes_level[1].size() << endl;
    // cout << nodes_level[1][3622] << endl;
    // if (merged_hnsw->linkLists_[62443] != nullptr) {
    //     cout << "yes" << endl;
    // }
    
    // DistanceGraph graph_test(1, M);
    // hnswlib::linklistsizeint* data_test = merged_hnsw->get_linklist(62443, 1);
    // if (data_test != nullptr) {
    //     cout << "yes" << endl;
    // }
    // int size_test = merged_hnsw->getListCount(data_test);
    // cout << size_test << endl;

    // hnswlib::tableint* datal_test = (hnswlib::tableint*) (data_test + 1);
    // for (int i = 0; i < size_test; i++) {
    //     cout << datal_test[i] << " ";
    // }
    // cout << endl;
    // for (int j = 0; j < size_test; j++) {
    //     float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(62443), merged_hnsw->getDataByInternalId(datal_test[j]), merged_hnsw->dist_func_param_);
    //     graph_test.addNeighbor(0, datal_test[j], dist);
    // }
    // graph_test.printGraph();
    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     cout << nodes_level[1][i] << " ";
    // }
    // cout << endl;

    // Level 1
    // No poping

    for (int level = merged_hnsw->maxlevel_ - 1; level <= merged_hnsw->maxlevel_; level++) {

        timer.reset();
        
        DistanceGraph graph_Ul(nodes_level[level].size(), M);

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];

            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
            int size = merged_hnsw->getListCount(data);
            // if (size == 0)  continue;
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int j = 0; j < size; j++) {
                float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
                graph_Ul.addNeighbor(i, datal[j], dist);
            }
        }

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];
            if (id < graph1ElementsNum) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnnTargetLayer(data + id * dim, 8, level);
                while(result.size() != 0) {
                    graph_Ul.addNeighbor(i, result.top().second + graph1ElementsNum, result.top().first);
                    result.pop();
                }
            }
            else {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnnTargetLayer(data + id * dim, 8, level);
                while(result.size() != 0) {
                    graph_Ul.addNeighbor(i, result.top().second, result.top().first);
                    result.pop();
                }
            }
        }

        cout << "Level " << level << " graph U1 and adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


        timer.reset();

        int iteration = 0;
        do {
            std::vector<std::unordered_set<int>> U(graph_Ul.getElementsNum());
            for (int i = 0; i < graph_Ul.getElementsNum(); i++) {
                auto pq = graph_Ul.getNeighbors(i);
                while (!pq.empty()) {
                    U[i].insert(pq.top().first);
                    for (int j = 0; j < nodes_level[level].size(); j++) {
                        if (nodes_level[level][j] == pq.top().first) {
                            U[j].insert(nodes_level[level][i]);
                            break;
                        }
                    }
                    // U[pq.top().first].insert(nodes_level[1][i]);
                    pq.pop();
                }
            }

            c = 0;
            for (int u = 0; u < graph_Ul.getElementsNum(); u++) 
            {
                std::vector<int> neighbors(U[u].begin(), U[u].end());
                for (int i = 0; i < neighbors.size(); i++) 
                {
                    for (int j = i + 1; j < neighbors.size(); j++) 
                    {
                        int si = neighbors[i];
                        int sj = neighbors[j];

                        if ((si < graph1ElementsNum && sj >= graph1ElementsNum) || 
                            (si >= graph1ElementsNum && sj < graph1ElementsNum))
                        {
                            float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

                            int si_idx = -1;     int sj_idx = -1;
                            for (int idx = 0; idx < nodes_level[level].size(); idx++) {
                                if (nodes_level[level][idx] == si) {
                                    si_idx = idx;
                                }
                                if (nodes_level[level][idx] == sj) {
                                    sj_idx = idx;
                                }
                            }

                            if (si_idx == -1 || sj_idx == -1) {
                                cout << u << endl;
                                cout << si << " " << sj << endl;
                                cout << "ERROR in finding index" << endl;
                            }

                            graph_Ul.updateNN(si_idx, sj, dist);
                            graph_Ul.updateNN(sj_idx, si, dist);
                        }
                    }
                }
            }

            iteration++;

        } while (c != 0 && iteration < 0);

        float loopTime = timer.getElapsedTimeSeconds();
        cout << "Level " << level << " looping time: " << loopTime << " seconds \t";
        cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;

        
        timer.reset();

        for (int i = 0; i < nodes_level[level].size(); i++) {
            int id = nodes_level[level][i];

            auto neighbors = graph_Ul.getNeighbors(i);
            int size = neighbors.size();
            hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, level);
            merged_hnsw->setListCount(data, size);
            hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
            for (int idx = 0; idx < size; idx++) {
                datal[idx] = neighbors.top().first;
                neighbors.pop();
            }
        }

        cout << "Level " << level << " neighors resetting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    }


    // timer.reset();

    // DistanceGraph graph_U1(nodes_level[1].size(), M);

    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     int id = nodes_level[1][i];

    //     hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
    //     int size = merged_hnsw->getListCount(data);
    //     // if (size == 0)  continue;
    //     hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //     for (int j = 0; j < size; j++) {
    //         float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(id), merged_hnsw->getDataByInternalId(datal[j]), merged_hnsw->dist_func_param_);
    //         graph_U1.addNeighbor(i, datal[j], dist);
    //     }
    // }
    // cout << "2" << endl;

    // graph_U1.printGraph(1, 1);

    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     int id = nodes_level[1][i];
    //     if (id < graph1ElementsNum) {
    //         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw2->searchKnnTargetLayer(data + id * dim, 4, 1);
    //         while(result.size() != 0) {
    //             graph_U1.addNeighbor(i, result.top().second + graph1ElementsNum, result.top().first);
    //             result.pop();
    //         }
    //     }
    //     else {
    //         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw1->searchKnnTargetLayer(data + id * dim, 4, 1);
    //         while(result.size() != 0) {
    //             graph_U1.addNeighbor(i, result.top().second, result.top().first);
    //             result.pop();
    //         }
    //     }
    // }
    // cout << "3" << endl;

    // graph_U1.printGraph(1, 1);

    // cout << "Level 1 graph U1 and adding time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;


    // timer.reset();

    // int iteration = 0;
    // do {
    //     std::vector<std::unordered_set<int>> U(graph_U1.getElementsNum());
    //     for (int i = 0; i < graph_U1.getElementsNum(); i++) {
    //         auto pq = graph_U1.getNeighbors(i);
    //         while (!pq.empty()) {
    //             U[i].insert(pq.top().first);
    //             // U[pq.top().first].insert(nodes_level[1][i]);
    //             pq.pop();
    //         }
    //     }
        
    //     cout << "4" << endl;

    //     c = 0;
    //     for (int u = 0; u < graph_U1.getElementsNum(); u++) 
    //     {
    //         std::vector<int> neighbors(U[u].begin(), U[u].end());
    //         for (int i = 0; i < neighbors.size(); i++) 
    //         {
    //             for (int j = i + 1; j < neighbors.size(); j++) 
    //             {
    //                 int si = neighbors[i];
    //                 int sj = neighbors[j];

    //                 if (si == sj) {
    //                     continue;
    //                 }

    //                 if ((si < graph1ElementsNum && sj >= graph1ElementsNum) || 
    //                     (si >= graph1ElementsNum && sj < graph1ElementsNum))
    //                 {
    //                     float dist = merged_hnsw->fstdistfunc_(merged_hnsw->getDataByInternalId(si), merged_hnsw->getDataByInternalId(sj), merged_hnsw->dist_func_param_);

    //                     int si_idx = -1;     int sj_idx = -1;
    //                     for (int idx = 0; idx < nodes_level[1].size(); idx++) {
    //                         if (nodes_level[1][idx] == si) {
    //                             si_idx = idx;
    //                         }
    //                         if (nodes_level[1][idx] == sj) {
    //                             sj_idx = idx;
    //                         }
    //                     }

    //                     if (si_idx == -1 || sj_idx == -1) {
    //                         cout << u << endl;
    //                         cout << si << " " << sj << endl;
    //                         cout << "ERROR in finding index" << endl;
    //                     }

    //                     graph_U1.updateNN(si_idx, sj, dist);
    //                     graph_U1.updateNN(sj_idx, si, dist);
    //                 }
    //             }
    //         }
    //     }
    //     iteration++;
    //     cout << "5" << endl;

    // } while (c != 0 && iteration < 0);

    // float loopTime = timer.getElapsedTimeSeconds();
    // cout << "Level 1 looping time: " << loopTime << " seconds \t";
    // cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;
    // cout << endl;


    // timer.reset();

    // for (int i = 0; i < nodes_level[1].size(); i++) {
    //     int id = nodes_level[1][i];

    //     auto neighbors = graph_U1.getNeighbors(i);
    //     int size = neighbors.size();
    //     hnswlib::linklistsizeint* data = merged_hnsw->get_linklist(id, 1);
    //     merged_hnsw->setListCount(data, size);
    //     hnswlib::tableint* datal = (hnswlib::tableint*) (data + 1);
    //     for (int idx = 0; idx < size; idx++) {
    //         datal[idx] = neighbors.top().first;
    //         neighbors.pop();
    //     }
    // }

    // cout << "Level 1 neighors resetting time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    // merged_hnsw->saveIndex("merged_hnsw_100k.bin");
    cout << endl;
    merged_hnsw->ef_ = ef_search;
    cout << "The ef search is " << ef_search << endl;

    // Compare search time and recall
    int correct = 0;
    timer.reset();

    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merged_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) {
            correct++;
        }
        // else {
        //     merged_hnsw->enterpoint_node_ = hnsw2->enterpoint_node_ + graph1ElementsNum;
        //     result = merged_hnsw->searchKnn(data + i * dim, 1);
        //     label = result.top().second;
        //     if (label == i) {
        //         correct++;
        //     }
        //     merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;
        // }
    }

    cout << "Merged graph search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / curElementsNum;
    std::cout << "Recall of merged graph: " << recall << "\n";
    cout << endl;

    delete merged_hnsw;
}