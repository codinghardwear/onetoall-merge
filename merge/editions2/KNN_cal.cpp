#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

const size_t DIM = 128;
const size_t NUM_POINTS = 100000;

vector<vector<float>> load_bvecs(const string& file_path, size_t num_points, size_t dim) {
    ifstream input(file_path, ios::binary);
    if (!input.is_open()) {
        throw runtime_error("Unable to open BVEC file: " + file_path);
    }

    vector<vector<float>> points(num_points, vector<float>(dim));
    for (size_t i = 0; i < num_points; i++) {
        unsigned char buffer[dim];
        int d = 0;
        input.read((char*)&d, sizeof(int)); 
        if (d != dim) {
            throw runtime_error("Dimension mismatch in BVEC file!");
        }
        input.read((char*)buffer, dim); 
        for (size_t j = 0; j < dim; j++) {
            points[i][j] = static_cast<float>(buffer[j]);
        }
    }
    return points;
}


float l2_distance(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < DIM; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


int main() {
    std::string base_path = "/media/raid5/myt/bigann_base.bvecs";    

    cout << "Loading " << NUM_POINTS << " points from SIFT1B dataset..." << endl;


    auto points = load_bvecs(base_path, NUM_POINTS, DIM);
    cout << "Loaded " << NUM_POINTS << " points successfully!" << endl;

    auto start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < NUM_POINTS; i++) {
        for (size_t j = NUM_POINTS; j >= (i + 1); j--) {
            float dist = l2_distance(points[i], points[j]);
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Time taken for 100k x 100k: " << duration.count() << " milliseconds" << endl;

    double estimated_time_1m = duration.count() * 100.0;
    cout << "Estimated time for 1M x 1M: " << estimated_time_1m / 1000.0 << " seconds." << endl;

    return 0;
}