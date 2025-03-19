#include "../hnsw/hnswlib/hnswlib.h"

#include <chrono>
#include <thread>

#include "timer.h"
#include "multi_thread.h"
#include "graph.h"

using std::cout;
using std::endl;

// Global c
int c;

// Construction parameters

int dim = 128;   int maxElementsNum = 100000;
int M = 16;     int ef_construction = 200;

hnswlib::L2Space space(dim);

hnswlib::HierarchicalNSW<float>* constructGraph1(int threadNum, float* data, int elementsNum) {

    hnswlib::HierarchicalNSW<float>* graph = new hnswlib::HierarchicalNSW<float>(&space, elementsNum, M, ef_construction);

    ParallelFor(0, elementsNum, threadNum, [&](size_t row, size_t threadId) {
        graph->addPoint((void*)(data + dim * row), row);
    });

    return graph;
}

void constructGraph2(hnswlib::HierarchicalNSW<float>* graph, int threadNum, float* data, int elementsNum) {
    ParallelFor(0, elementsNum, threadNum, [&](size_t row, size_t threadId) {
        graph->addPoint((void*)(data + dim * row), row);
    });
}

int main() {

    size_t graph1ElementsNum = maxElementsNum/2;    size_t graph2ElementsNum = maxElementsNum/2;
    size_t curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    // Initing index
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, graph1ElementsNum, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, graph2ElementsNum, M, ef_construction);

    // Generate random data
    std::uniform_real_distribution<> distrib_real;
    
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * curElementsNum];
    for (size_t i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }

    float* data1 = new float[dim * graph1ElementsNum];
    for (size_t i = 0; i < dim * graph1ElementsNum; i++) {
        data1[i] = data[i]; 
    }

    float* data2 = new float[dim * graph2ElementsNum];
    for (size_t i = dim * graph1ElementsNum; i < dim * curElementsNum; i++) {
        data2[i - dim * graph1ElementsNum] = data[i];
    }


    // cout << "Data preparation test:" << endl;
    // cout << data[0] << " " << data1[0] << endl;
    // cout << data[dim*graph1ElementsNum] << " " << data2[0] << endl;
    // cout << endl;

    int threadNum = 12;

    // Generate graph
    Timer timer;        timer.reset();

    // std::thread thread1(constructGraph2, alg_hnsw1, threadNum / 2, data1, graph1ElementsNum);
    // std::thread thread2(constructGraph2, alg_hnsw2, threadNum / 2, data2, graph2ElementsNum);
    // thread1.join();
    // thread2.join();

    // hnswlib::HierarchicalNSW<float>* alg_hnsw1 = constructGraph(threadNum, data1, graph1ElementsNum);
    // hnswlib::HierarchicalNSW<float>* alg_hnsw2 = constructGraph(threadNum, data2, graph2ElementsNum);

    // for (size_t i = 0; i < graph1ElementsNum; i++) 
    //     alg_hnsw1->addPoint(data1 + i * dim, i);

    // for (size_t i = 0; i < graph2ElementsNum; i++) 
    //     alg_hnsw2->addPoint(data2 + i * dim, i);
    
    // Save index
    std::string hnsw_path1 = "hnsw1_5w.bin";
    std::string hnsw_path2 = "hnsw2_5w.bin";

    // alg_hnsw1->saveIndex(hnsw_path1);
    // alg_hnsw2->saveIndex(hnsw_path2);
    // delete alg_hnsw1;
    // delete alg_hnsw2;

    // Load index
    // timer.reset();
    cout << "graph1" << endl;
    alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path1);
    cout << "graph2" << endl;
    alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path2);
    // cout << "Reloading time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // cout << endl;

    // cout << alg_hnsw1->enterpoint_node_ << endl;
    // cout << alg_hnsw2->enterpoint_node_ << endl;
    // cout << endl;

    // Test on computation

    // cout << "Data comparison:" << endl;
    // cout << data1[0] << "\t"  << *((float*) alg_hnsw1->getDataByInternalId(0)) << endl;
    // cout << data1[50] << "\t" << *((float*) alg_hnsw1->getDataByInternalId(50)) << endl;
    // cout << data1[99] << "\t" << *((float*) alg_hnsw1->getDataByInternalId(99)) << endl;
    // cout << data2[0] << "\t"  << *((float*) alg_hnsw2->getDataByInternalId(0)) << endl;
    // cout << data2[50] << "\t" << *((float*) alg_hnsw2->getDataByInternalId(50)) << endl;
    // cout << data2[99] << "\t" << *((float*) alg_hnsw2->getDataByInternalId(99)) << endl;

    // cout << "Computation Check:" << endl;
    // float dis1 = alg_hnsw1->fstdistfunc_(alg_hnsw1->getDataByInternalId(0), alg_hnsw1->getDataByInternalId(1), alg_hnsw1->dist_func_param_);
    // cout << dis1 << "\t";
    // float t1 = data1[0] - data1[1];
    // cout << t1 * t1 << endl;

    // float dis2 = alg_hnsw2->fstdistfunc_(alg_hnsw2->getDataByInternalId(0), alg_hnsw2->getDataByInternalId(1), alg_hnsw2->dist_func_param_);
    // cout << dis2 << "\t";
    // float t2 = data2[0] - data2[1];
    // cout << t2 * t2 << endl;


    // Step 1: cut off 3/4 M (here is 12) nodes from the graph
    // If number of neighbors <= 5/4 M (here is 20), don't perform cutoff

    // line = 16, cutoff = 8

    size_t line = M;     size_t cutoff = M * 0.5;
    // cout << "cutoff line is " << line << " and cutoff is " << cutoff << endl;
    cout << "Add " << cutoff << endl; 
    // cout << endl;
    
    timer.reset();

    Graph graph1_p(graph1ElementsNum);
    for (size_t id = 0; id < graph1ElementsNum; id++)
    {
        int* data = (int*) alg_hnsw1->get_linklist0(id);
        size_t size = alg_hnsw1->getListCount((hnswlib::linklistsizeint*) data);

        for (size_t j = 1; j <= size; j++) {
            int neighbor_id = *(data + j);
            graph1_p.addNeighbor(id, neighbor_id);
        }
    }

    Graph graph2_p(graph2ElementsNum);
    for (size_t id = 0; id < graph2ElementsNum; id++)
    {
        int* data = (int*) alg_hnsw2->get_linklist0(id);
        size_t size = alg_hnsw2->getListCount((hnswlib::linklistsizeint*) data);

        for (size_t j = 1; j <= size; j++) {
            int neighbor_id = *(data + j);
            graph2_p.addNeighbor(id, neighbor_id+graph1ElementsNum);
        }
    }

    for (size_t id = 0; id < graph1ElementsNum; id++) {
        graph1_p.appendRandomNeighbors(id, graph1ElementsNum, curElementsNum-1, cutoff);
    }

    for (size_t id = 0; id < graph2ElementsNum; id++) {
        graph2_p.appendRandomNeighbors(id, 0, graph1ElementsNum-1, cutoff);
    }

    float dist = 0;
    DistanceGraph graph_U(curElementsNum, 2*M);

    for (size_t graph1_id = 0; graph1_id < graph1ElementsNum; graph1_id++)
    {
        for (size_t neighbor : graph1_p.getNeigbors(graph1_id))
        {
            if (neighbor < graph1ElementsNum)
                dist = alg_hnsw1->fstdistfunc_(alg_hnsw1->getDataByInternalId(graph1_id), alg_hnsw1->getDataByInternalId(neighbor), alg_hnsw1->dist_func_param_);
            else
                dist = alg_hnsw1->fstdistfunc_(alg_hnsw1->getDataByInternalId(graph1_id), alg_hnsw2->getDataByInternalId(neighbor-graph1ElementsNum), alg_hnsw1->dist_func_param_);
            graph_U.addNeighbor(graph1_id, neighbor, dist);
        }
    }

    for (size_t graph2_id = 0; graph2_id < graph2ElementsNum; graph2_id++)
    {
        for (size_t neighbor : graph2_p.getNeigbors(graph2_id))
        {
            if (neighbor < graph1ElementsNum)
                dist = alg_hnsw2->fstdistfunc_(alg_hnsw2->getDataByInternalId(graph2_id), alg_hnsw1->getDataByInternalId(neighbor), alg_hnsw2->dist_func_param_);
            else
                dist = alg_hnsw2->fstdistfunc_(alg_hnsw2->getDataByInternalId(graph2_id), alg_hnsw2->getDataByInternalId(neighbor-graph1ElementsNum), alg_hnsw2->dist_func_param_);
            graph_U.addNeighbor(graph2_id+graph1ElementsNum, neighbor, dist);
        }
    }

    cout << "Graph U time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;

    graph_U.printGraph(100, 100);

    // Loop starts

    timer.reset();

    int iteration = 0;
    do {
        std::vector<std::unordered_set<size_t>> U(graph_U.getElementsNum());
        for (size_t i = 0; i < graph_U.getElementsNum(); i++) {
            auto pq = graph_U.getNeighbors(i);
            while (!pq.empty()) {
                U[i].insert(pq.top().first);
                U[pq.top().first].insert(i);
                pq.pop();
            }
        }

        c = 0;
        for (size_t u = 0; u < graph_U.getElementsNum(); u++) 
        {
            std::vector<size_t> neighbors(U[u].begin(), U[u].end());
            for (size_t i = 0; i < neighbors.size(); i++) 
            {
                for (size_t j = i + 1; j < neighbors.size(); j++) 
                {
                    size_t si = neighbors[i];
                    size_t sj = neighbors[j];
                    if ((si < graph1ElementsNum && sj >= graph1ElementsNum) || 
                        (si >= graph1ElementsNum && sj < graph1ElementsNum))
                    {
                        float dist;
                        if (si < graph1ElementsNum && sj >= graph1ElementsNum)
                            dist = alg_hnsw1->fstdistfunc_(alg_hnsw1->getDataByInternalId(si), alg_hnsw2->getDataByInternalId(sj-graph1ElementsNum), alg_hnsw1->dist_func_param_);
                        else
                            dist = alg_hnsw1->fstdistfunc_(alg_hnsw2->getDataByInternalId(si-graph1ElementsNum), alg_hnsw1->getDataByInternalId(sj), alg_hnsw1->dist_func_param_);

                        graph_U.updateNN(si, sj, dist);
                        graph_U.updateNN(sj, si, dist);
                    }
                }
            }
        }
        iteration++;

    } while (c != 0 && iteration < 0);

    float loopTime = timer.getElapsedTimeSeconds();
    cout << "Loop time: " << loopTime << " seconds" << endl;
    cout << iteration << " iterations with " << (float)loopTime / iteration << " average time" << endl;
    // cout << "No reverse" << endl;
    // cout << endl;

    // Save merged index

    // timer.reset();
    hnswlib::HierarchicalNSW<float>* merged_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    // hnswlib::HierarchicalNSW<float>* merged_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, maxElementsNum, M, ef_construction);
    merged_hnsw1->cur_element_count = curElementsNum;
    merged_hnsw1->maxlevel_ = alg_hnsw1->maxlevel_;
    merged_hnsw1->enterpoint_node_ = alg_hnsw1->enterpoint_node_;

    // Level 0
    for (size_t id = 0; id < curElementsNum; id++)
    {
        memset(merged_hnsw1->data_level0_memory_ + id * merged_hnsw1->size_data_per_element_ + merged_hnsw1->offsetLevel0_, 0, merged_hnsw1->size_data_per_element_);

        // Set data and label
        if (id < graph1ElementsNum)
        {
            memcpy(merged_hnsw1->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw1->getDataByInternalId(id), alg_hnsw1->getDataByInternalId(id), merged_hnsw1->data_size_);
        }
        else
        {
            memcpy(merged_hnsw1->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw1->getDataByInternalId(id), alg_hnsw2->getDataByInternalId(id-graph1ElementsNum), merged_hnsw1->data_size_);
        }

        // Set neighbors
        auto neighbors = graph_U.getNeighbors(id);
        size_t neighborSize = neighbors.size();
        // std::vector<size_t> neighborsClosest(neighborSize);
        // for (int i = neighborSize-1; i >= 0; i--)
        // {
        //     neighborsClosest[i] = neighbors.top().first;
        //     neighbors.pop();
        // }

        hnswlib::linklistsizeint* neighborSizePtr = merged_hnsw1->get_linklist0(id);
        merged_hnsw1->setListCount(neighborSizePtr, neighborSize);
        hnswlib::tableint* neighborIdPtr = (hnswlib::tableint*) (neighborSizePtr + 1);
        for (size_t idx = 0; idx < neighborSize; idx++) {
            neighborIdPtr[idx] = neighbors.top().first;
            neighbors.pop();
        }
            // neighborIdPtr[idx] = neighborsClosest[idx];
    }

    // DistanceGraph merged_graph(curElementsNum, 2*M);
    // for (size_t id = 0; id < curElementsNum; id++)
    // {
    //     int* data = (int*) merged_hnsw1->get_linklist0(id);
    //     size_t size = merged_hnsw1->getListCount((hnswlib::linklistsizeint*) data);

    //     for (size_t j = 1; j <= size; j++) 
    //     {
    //         int neighbor_id = *(data + j);
    //         float dist = merged_hnsw1->fstdistfunc_(merged_hnsw1->getDataByInternalId(id), merged_hnsw1->getDataByInternalId(neighbor_id), merged_hnsw1->dist_func_param_);
    //         merged_graph.addNeighbor(id, neighbor_id, dist);
    //     }
    // }
    // merged_graph.printGraph(100, 100);
    // cout << endl;

    // Other levels

    // Copy graph 1 data
    for (size_t id = 0; id < graph1ElementsNum; id++)
    {
        merged_hnsw1->element_levels_[id] = alg_hnsw1->element_levels_[id];
        if (merged_hnsw1->element_levels_[id] == 0)
            merged_hnsw1->linkLists_[id] = nullptr;
        else
        {
            size_t size = merged_hnsw1->size_links_per_element_ * merged_hnsw1->element_levels_[id];
            merged_hnsw1->linkLists_[id] = (char*) malloc(size);

            if (merged_hnsw1->linkLists_[id] != nullptr) {
                memset(merged_hnsw1->linkLists_[id], 0, size);
                memcpy(merged_hnsw1->linkLists_[id], alg_hnsw1->linkLists_[id], size);
            }
            // else {
            //     cout << "no enough space";
            // }
        }
    }

    // Copy graph 2 data
    for (size_t id = 0; id < graph2ElementsNum; id++)
    {
        merged_hnsw1->element_levels_[id+graph1ElementsNum] = alg_hnsw2->element_levels_[id];
        if (merged_hnsw1->element_levels_[id+graph1ElementsNum] == 0)
            merged_hnsw1->linkLists_[id+graph1ElementsNum] = nullptr;
        else
        {
            size_t size = merged_hnsw1->size_links_per_element_ * merged_hnsw1->element_levels_[id+graph1ElementsNum];
            merged_hnsw1->linkLists_[id+graph1ElementsNum] = (char*) malloc(size);

            if (merged_hnsw1->linkLists_[id] != nullptr) {
                memset(merged_hnsw1->linkLists_[id+graph1ElementsNum], 0, size);
                memcpy(merged_hnsw1->linkLists_[id+graph1ElementsNum], alg_hnsw2->linkLists_[id], size);     
            }
            // else {
            //     cout << "no enough space";
            // }

        }
    }
    // cout << "Save merged index time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // cout << endl;

    std::string merged_hnsw1_path = "merged_hnsw1.bin";
    merged_hnsw1->saveIndex(merged_hnsw1_path);
    delete merged_hnsw1;
    merged_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, merged_hnsw1_path);
    // cout << merged_hnsw1->enterpoint_node_ << endl;

    // std::random_device rd;  std::mt19937 gen(rd());
    // std::uniform_int_distribution<> dis(graph1ElementsNum, curElementsNum-1);
    // int random_number = dis(gen);

    // merged_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, merged_hnsw1_path);
    // merged_hnsw2->enterpoint_node_ = random_number;
    // merged_hnsw2->enterpoint_node_ = alg_hnsw2->enterpoint_node_ + graph1ElementsNum;
    // cout << merged_hnsw2->enterpoint_node_ << endl;
    // cout << endl;
    
   // Compare search time and recall
    // int correct = 0;
    // timer.reset();
    // for (int i = 0; i < curElementsNum; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // cout << "Original search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // float recall = (float)correct / curElementsNum;
    // std::cout << "Recall of original graph: " << recall << "\n";
    // cout << endl;

    cout << "Total time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    int correct = 0;
    timer.reset();
    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result1 = merged_hnsw1->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label1 = result1.top().second;
        if (label1 == i) correct++;
        // else
        // {
        //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result2 = merged_hnsw2->searchKnn(data + i * dim, 1);
        //     hnswlib::labeltype label2 = result2.top().second; 
        //     if (label2 == i) correct++;
        // }
    }
    cout << "Merged search time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    float recall = (float)correct / curElementsNum;
    std::cout << "Recall of merged graph: " << recall << "\n";
    cout << endl;
    
    
    delete[] data;
    delete[] data1;
    delete[] data2;
    delete alg_hnsw;
    delete alg_hnsw1;
    delete alg_hnsw2;
    delete merged_hnsw1;
    return 0;
}
