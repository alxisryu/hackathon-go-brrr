#ifndef LOADER_H
#define LOADER_H

#include <vector>
#include <string>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

struct Layer {
    Matrix weights;
    Vector biases;
};

void loadWeightsAndBiases(const std::string& filename, std::vector<Layer>& layers);

#endif // LOADER_H
