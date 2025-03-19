#ifndef GRAPH_H
#define GRAPH_H

#include <cstdlib>
#include <ctime>

extern int c;

class Graph {

    int verticesNum;
    std::vector<std::list<int>> adjLists;

public:
    Graph(int elementsNum) {
        verticesNum = elementsNum;
        adjLists.resize(verticesNum);
    }

    ~Graph() {

    }

    void addNeighbor(int src, int dest) {
        adjLists[src].push_back(dest);
    }

    const std::list<int>& getNeigbors(int vertex) const {
        return adjLists[vertex];
    }

    void appendRandomNeighbors(int vertex, int start, int end, int neighborsNum) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(start, end);
        // std::srand(static_cast<unsigned int>(std::time(nullptr)));
        std::unordered_set<int> set;

        int count = 0;
        while (count < neighborsNum) {
            int rand = dis(gen);
            // int random = start + rand() % (end - start + 1);
            if (set.find(rand) == set.end()) {
                addNeighbor(vertex, rand);
                set.insert(rand);
                count++;
            }
        }
    }

    // Print full graph
    void printGraph() {
        for (int i = 0; i < verticesNum; i++) {
            cout << "Vertex " << i << ":";
            for (auto v : adjLists[i]) {
                cout << " -> " << v;
            }
            cout << endl;
        }
    }

    // Print graph with vertices [start, end]
    void printGraph(int start, int end) {
        for (int i = start; i <= end; i++) {
            cout << "Vertex " << i << ":";
            for (auto v : adjLists[i]){
                cout << " -> " << v;
            }
            cout << endl;
        }
    }
};

struct compare {
    bool operator()(const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    }
};

class DistanceGraph {

    int verticesNum;
    int maxNeighborsNum;   // IMPORTANT!!!
    std::vector<std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, compare>> adjLists;   // <NodeID, Distance> pair
    std::vector<std::unordered_set<int>> neighborSets;

public:
    DistanceGraph(int elementsNum, int maxNeighbors) {
        verticesNum = elementsNum;
        maxNeighborsNum = maxNeighbors;
        adjLists.resize(verticesNum);
        neighborSets.resize(verticesNum);
    }

    ~DistanceGraph() {

    }

    int getElementsNum() {
        return verticesNum;
    }

    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, compare>& getNeighbors(int vertex) {
        return adjLists[vertex];
    }

    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, compare> getNeighborsCopy(int vertex) const {
        return adjLists[vertex];
    }

    void addNeighbor(int src, int dest, float dist) {
        if (neighborSets[src].find(dest) == neighborSets[src].end()) {
            if (adjLists[src].size() < maxNeighborsNum) {
                adjLists[src].emplace(dest, dist);
                neighborSets[src].insert(dest);
                c++;
                // cout << c << " ";
            } 
            else if (dist < adjLists[src].top().second) {
                neighborSets[src].erase(adjLists[src].top().first);
                adjLists[src].pop();
                adjLists[src].emplace(dest, dist);
                neighborSets[src].insert(dest);
                c++;
                // cout << c << " ";
            }
        }
    }

    DistanceGraph reverseGraph() {
        DistanceGraph reversedGraph(verticesNum, maxNeighborsNum);
        for (int i = 0; i < verticesNum; i++) {
            auto pq = adjLists[i];
            while (!pq.empty()) {
                reversedGraph.addNeighbor(pq.top().first, i, pq.top().second);
                pq.pop();
            }
        }
        return reversedGraph;
    }

    // DistanceGraph* reverseGraph() {
    //     DistanceGraph* reversedGraph = new DistanceGraph(verticesNum, maxNeighborsNum);
    //     for (int i = 0; i < verticesNum; i++) {
    //         auto pq = adjLists[i];
    //         while (!pq.empty()) {
    //             reversedGraph->addNeighbor(pq.top().first, i, pq.top().second);
    //             pq.pop();
    //         }
    //     }
    //     return reversedGraph;
    // }

    void updateNN(int vertex, int neighbor, float distance) {
        addNeighbor(vertex, neighbor, distance);
    }

    // Print full graph
    void printGraph() {
        for (int i = 0; i < verticesNum; i++) {
            cout << "Vertex " << i << ":";
            auto pq = adjLists[i];
            while (!pq.empty()) {
                cout << " -> " << pq.top().first << " (dist: " << pq.top().second << ")";
                pq.pop();
            }
            cout << endl;   cout << endl;
        }
    }

    // Print graph with vertices [start, end]
    void printGraph(int start, int end) {
        for (int i = start; i <= end; i++) {
            cout << "Vertex " << i << ":";
            auto pq = adjLists[i];
            while (!pq.empty()) {
                cout << " -> " << pq.top().first << " (dist: " << pq.top().second << ")";
                pq.pop();
            }
            cout << endl;   cout << endl;
        }
    }
};

#endif