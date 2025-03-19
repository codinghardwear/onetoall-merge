#ifndef SMERGE_H
#define SMERGE_H

#include <random>
#include "../hnsw/hnswlib/hnswlib.h"
#include "timer.h"
#include "graph.h"


class SymmetricMerge
{
public:
    int dim_;
    int M_;
    int ef_construction_;
    int line_;
    int cutoff_;
    int c_;
    hnswlib::L2Space* space_;
    hnswlib::HierarchicalNSW<float>* hnsw1_;
    hnswlib::HierarchicalNSW<float>* hnsw2_;

    SymmetricMerge(int dim, int M, int ef_construction, int line, int cutoff, int c, hnswlib::L2Space* space,
                   hnswlib::HierarchicalNSW<float>* hnsw1,
                   hnswlib::HierarchicalNSW<float>* hnsw2)
    {
        dim_ = dim;
        M_ = M;
        ef_construction_ = ef_construction;
        line_ = line;
        cutoff_ = cutoff;
        space_ = space;
        
    }

}

extern Timer timer;

float* generateData(int num, int seed)
{
    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  
    rng.seed(seed);

    float* data = new float[dim * num];
    for (int i = 0; i < dim * num; i++)
        data[i] = distrib_real(rng);

    return data;
}

hnswlib::HierarchicalNSW<float>* build_subgraph(hnswlib::L2Space* space, float* data, int elementsNum)
{
    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(space, elementsNum, M, ef_construction);

    for (int i = 0; i < elementsNum; i++)
        hnsw->addPoint(data + i * dim, i);

    return hnsw;
}

std::pair<Graph*, DistanceGraph*> prepare_graphAB(hnswlib::HierarchicalNSW<float>* hnsw, char graph_id, int graphAElementsNum)
{
    // Step 1: cut off 3/4 M (here is 12) nodes from the graph
    // If number of neighbors <= 5/4 M (here is 20), don't perform cutoff
    int elementsNum = hnsw->cur_element_count;

    Graph* graph_p = new Graph(elementsNum);    
    DistanceGraph* graph_n = new DistanceGraph(elementsNum, 2 * M);

    for (int id = 0; id < elementsNum; id++)
    {
        int* data = (int*) hnsw->get_linklist0(id);
        size_t size = hnsw->getListCount((hnswlib::linklistsizeint*) data);

        if (size <= line)
        {
            for (int j = 1; j <= size; j++) {
                int neighbor_id = *(data + j);
                if (graph_id == 'A')
                    graph_p->addNeighbor(id, neighbor_id);
                else if (graph_id == 'B')
                    graph_p->addNeighbor(id, neighbor_id + graphAElementsNum);
            }
        }

        else
        {
            for (int j = 1; j <= size - cutoff; j++) {
                int neighbor_id = *(data + j);
                if (graph_id == 'A')
                    graph_p->addNeighbor(id, neighbor_id);
                else if (graph_id == 'B')
                    graph_p->addNeighbor(id, neighbor_id + graphAElementsNum);
            }
            for (int j = size - cutoff + 1; j <= size; j++) {
                int neighbor_id = *(data + j);
                float dist = hnsw->fstdistfunc_(hnsw->getDataByInternalId(id), hnsw->getDataByInternalId(neighbor_id), hnsw->dist_func_param_);
                if (graph_id == 'A')
                    graph_n->addNeighbor(id, neighbor_id, dist);
                else if (graph_id == 'B')
                    graph_n->addNeighbor(id, neighbor_id + graphAElementsNum, dist);              
            }
        }
    }

    return std::make_pair(graph_p, graph_n);
}

std::tuple<DistanceGraph*, DistanceGraph*, DistanceGraph*> prepare_graphU(hnswlib::HierarchicalNSW<float>* hnsw1, 
                                                                         hnswlib::HierarchicalNSW<float>* hnsw2)
{
    int graphAElementsNum = hnsw1->cur_element_count;   
    int graphBElementsNum = hnsw2->cur_element_count;

    // Step 1:
    auto [graphA_p, graphA_n] = prepare_graphAB(hnsw1, 'A', graphAElementsNum);
    auto [graphB_p, graphB_n] = prepare_graphAB(hnsw2, 'B', graphAElementsNum);

    graphA_p->printGraph();
    
    // Step 2: randomly add 3/4 M (here is 12) nodes from another set
    int curElementsNum = graphAElementsNum + graphBElementsNum;

    for (int id = 0; id < graphAElementsNum; id++)
        graphA_p->appendRandomNeighbors(id, graphAElementsNum, curElementsNum-1, cutoff);
    for (int id = 0; id < graphBElementsNum; id++)
        graphB_p->appendRandomNeighbors(id, 0, graphAElementsNum-1, cutoff);

    // Step 3: Union graph 1 and graph 2
    float dist = 0;
    DistanceGraph* graphU = new DistanceGraph(curElementsNum, 2 * M);

    for (int graphA_id = 0; graphA_id < graphAElementsNum; graphA_id++)
    {
        for (int neighbor : graphA_p->getNeigbors(graphA_id))
        {
            if (neighbor < graphAElementsNum)
                dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(graphA_id), hnsw1->getDataByInternalId(neighbor), hnsw1->dist_func_param_);
            else
                dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(graphA_id), hnsw2->getDataByInternalId(neighbor - graphAElementsNum), hnsw1->dist_func_param_);
            graphU->addNeighbor(graphA_id, neighbor, dist);
        }
    }
    for (int graphB_id = 0; graphB_id < graphBElementsNum; graphB_id++)
    {
        for (int neighbor : graphB_p->getNeigbors(graphB_id))
        {
            if (neighbor < graphAElementsNum)
                dist = hnsw2->fstdistfunc_(hnsw2->getDataByInternalId(graphB_id), hnsw1->getDataByInternalId(neighbor), hnsw2->dist_func_param_);
            else
                dist = hnsw2->fstdistfunc_(hnsw2->getDataByInternalId(graphB_id), hnsw2->getDataByInternalId(neighbor - graphAElementsNum), hnsw2->dist_func_param_);
            graphU->addNeighbor(graphB_id + graphAElementsNum, neighbor, dist);
        }
    }

    delete graphA_p;
    delete graphB_p;

    return std::make_tuple(graphU, graphA_n, graphB_n);
}                                   

hnswlib::HierarchicalNSW<float>* build_merged_graph(hnswlib::L2Space* space,
                                                    hnswlib::HierarchicalNSW<float>* hnsw1, 
                                                    hnswlib::HierarchicalNSW<float>* hnsw2,
                                                    DistanceGraph* graphU)
{
    int graphAElementsNum = hnsw1->cur_element_count;   int graphBElementsNum = hnsw2->cur_element_count;
    int curElementsNum = graphAElementsNum + graphBElementsNum;
    hnswlib::HierarchicalNSW<float>* merged_hnsw = new hnswlib::HierarchicalNSW<float>(space, curElementsNum, M, ef_construction);

    // Set basic parameters
    merged_hnsw->cur_element_count = curElementsNum;
    merged_hnsw->maxlevel_ = hnsw1->maxlevel_;
    merged_hnsw->enterpoint_node_ = hnsw1->enterpoint_node_;

    // Set level 0
    for (size_t id = 0; id < curElementsNum; id++)
    {
        memset(merged_hnsw->data_level0_memory_ + id * merged_hnsw->size_data_per_element_ + merged_hnsw->offsetLevel0_, 0, merged_hnsw->size_data_per_element_);

        // Set data and label
        if (id < hnsw1->cur_element_count)
        {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), hnsw1->getDataByInternalId(id), merged_hnsw->data_size_);
        }
        else
        {
            memcpy(merged_hnsw->getExternalLabeLp(id), &id, sizeof(hnswlib::labeltype));
            memcpy(merged_hnsw->getDataByInternalId(id), hnsw2->getDataByInternalId(id - hnsw1->cur_element_count), merged_hnsw->data_size_);
        }

        // Set neighbors
        auto neighbors = graphU->getNeigbors(id);
        size_t neighborSize = neighbors.size();
        std::vector<size_t> neighborsClosest(neighborSize);
        for (int i = neighborSize-1; i >= 0; i--)
        {
            neighborsClosest[i] = neighbors.top().first;
            neighbors.pop();
        }

        hnswlib::linklistsizeint* neighborSizePtr = merged_hnsw->get_linklist0(id);
        merged_hnsw->setListCount(neighborSizePtr, M);
        hnswlib::tableint* neighborIdPtr = (hnswlib::tableint*) (neighborSizePtr + 1);
        for (int idx = 0; idx < M; idx++)
            neighborIdPtr[idx] = neighborsClosest[idx];
    }

    // Set other levels
    // Copy graph A data
    for (size_t id = 0; id < graphAElementsNum; id++)
    {
        merged_hnsw->element_levels_[id] = hnsw1->element_levels_[id];
        if (merged_hnsw->element_levels_[id] == 0)
            merged_hnsw->linkLists_[id] = nullptr;
        else
        {
            size_t size = merged_hnsw->size_links_per_element_ * merged_hnsw->element_levels_[id];
            merged_hnsw->linkLists_[id] = (char*) malloc(size);
            memset(merged_hnsw->linkLists_[id], 0, size);
            memcpy(merged_hnsw->linkLists_[id], hnsw1->linkLists_[id], size);
        }
    }

    // Copy graph B data
    for (size_t id = 0; id < graphBElementsNum; id++)
    {
        merged_hnsw->element_levels_[id + graphAElementsNum] = hnsw2->element_levels_[id];
        if (merged_hnsw->element_levels_[id + graphAElementsNum] == 0)
            merged_hnsw->linkLists_[id + graphAElementsNum] = nullptr;
        else
        {
            size_t size = merged_hnsw->size_links_per_element_ * merged_hnsw->element_levels_[id + graphAElementsNum];
            merged_hnsw->linkLists_[id + graphAElementsNum] = (char*) malloc(size);
            memset(merged_hnsw->linkLists_[id + graphAElementsNum], 0, size);
            memcpy(merged_hnsw->linkLists_[id + graphAElementsNum], hnsw2->linkLists_[id], size);
        }
    }

    delete graphU;

    return merged_hnsw;
}


hnswlib::HierarchicalNSW<float>* s_merge(hnswlib::L2Space* space,
                                         hnswlib::HierarchicalNSW<float>* hnsw1, 
                                         hnswlib::HierarchicalNSW<float>* hnsw2,
                                         float* data1,
                                         float* data2)
{
    int graphAElementsNum = hnsw1->cur_element_count;   int graphBElementsNum = hnsw2->cur_element_count;
    auto [graphU, graphA_n, graphB_n] = prepare_graphU(hnsw1, hnsw2);

    do {
        // Step 4: Union graph U with reversed graph U
        // timer.reset();
        DistanceGraph* graphR = graphU->reverseGraph();
        std::vector<std::unordered_set<size_t>> U(graphU->getElementsNum());

        for (int i = 0; i < graphU->getElementsNum(); i++) 
        {
            auto pq = graphU->getNeigbors(i);
            while (!pq.empty()) 
            {
                U[i].insert(pq.top().first);
                pq.pop();
            }
            pq = graphR->getNeigbors(i);
            while (!pq.empty()) 
            {
                U[i].insert(pq.top().first);
                pq.pop();
            }
        }

        delete graphR;
        // cout << "Step4 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
        // cout << endl;

        // Step 5: update neighbors
        // timer.reset();
        c = 0;
        for (size_t u = 0; u < graphU->getElementsNum(); u++) 
        {
            std::vector<size_t> neighbors(U[u].begin(), U[u].end());
            for (int i = 0; i < neighbors.size(); i++) 
            {
                for (int j = i + 1; j < neighbors.size(); j++) 
                {
                    size_t si = neighbors[i];
                    size_t sj = neighbors[j];
                    if ((si < graphAElementsNum && sj >= graphAElementsNum) || 
                        (si >= graphAElementsNum && sj < graphAElementsNum))
                    {
                        float dist;
                        if (si < graphAElementsNum && sj >= graphAElementsNum)
                            dist = hnsw1->fstdistfunc_(hnsw1->getDataByInternalId(si), hnsw2->getDataByInternalId(sj - graphAElementsNum), hnsw1->dist_func_param_);
                        else
                            dist = hnsw1->fstdistfunc_(hnsw2->getDataByInternalId(si - graphAElementsNum), hnsw1->getDataByInternalId(sj), hnsw1->dist_func_param_);

                        graphU->updateNN(si, sj, dist);
                        graphU->updateNN(sj, si, dist);
                    }
                }
            }
        }
        // cout << "Step5 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
        // cout << endl;

    } while (c != 0);

    // Step 6: Final merge
    // timer.reset();
    for (int i = 0; i < graphAElementsNum; i++) 
    {
        auto edges = graphA_n->getNeigbors(i);
        while (!edges.empty()) 
        {
            auto edge = edges.top();
            graphU->addNeighbor(i, edge.first, edge.second);
            edges.pop();
        }
    }

    for (int i = 0; i < graphBElementsNum; i++) 
    {
        auto edges = graphB_n->getNeigbors(i);
        while (!edges.empty()) 
        {
            auto edge = edges.top();
            graphU->addNeighbor(i+graphAElementsNum, edge.first, edge.second);
            edges.pop();
        }
    }

    delete graphA_n;    
    delete graphB_n;

    // cout << "Step6 time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    // cout << endl;
    // graphU->printGraph();
    // cout << endl;

    // Step7: Build merged hnsw graph
    // timer.reset();
    hnswlib::HierarchicalNSW<float>* merged_hnsw = build_merged_graph(space, hnsw1, hnsw2, graphU);
    return merged_hnsw;
}                                


#endif