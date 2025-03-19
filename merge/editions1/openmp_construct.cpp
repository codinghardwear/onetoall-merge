#include "../hnsw/hnswlib/hnswlib.h"

#include <chrono>
#include <thread>
#include <algorithm>
#include <omp.h>

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

Timer timer;

int main() {
    int graph1ElementsNum = maxElementsNum / 2;    int graph2ElementsNum = maxElementsNum / 2;
    int curElementsNum = graph1ElementsNum + graph2ElementsNum;

    cout << curElementsNum << " elements with " << dim << " dimensions" << endl;
    cout << endl;

    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, curElementsNum, M, ef_construction);

    std::uniform_real_distribution<> distrib_real;
    std::mt19937 rng;  rng.seed(0);
    float* data = new float[dim * curElementsNum];
    for (int i = 0; i < dim * curElementsNum; i++) {
        data[i] = distrib_real(rng);  
    }

    timer.reset();

    #pragma omp parallel for 
    for (int i = 0; i < curElementsNum; i++) {
        hnsw->addPoint(data + i * dim, i);
    }

    cout << "openMP complete hnsw construction time: " << timer.getElapsedTimeSeconds() << " seconds" << endl;
    cout << endl;

    std::vector<hnswlib::labeltype> result_record(curElementsNum);
    timer.reset();

    #pragma omp parallel for
    for (int i = 0; i < curElementsNum; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw->searchKnn(data + i * dim, 1);
        result_record[i] = result.top().second;
    }

    int correct = 0;
    for (int i = 0; i < curElementsNum; i++) {
        if (result_record[i] == i) {
            correct++;
        }
    }

    cout << "openMP complete hnsw search time: " << timer.getElapsedTimeSeconds() << " seconds \t";
    float recall = (float)correct / curElementsNum;     std::cout << "Recall of complete hnsw graph: " << recall << "\n";
    cout << endl;
}