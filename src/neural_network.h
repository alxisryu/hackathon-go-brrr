#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

struct Layer {
    Matrix weights;
    Vector biases;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<Layer>& layers);
    Vector forward(const Vector& input);

private:
    std::vector<Layer> layers_;
    Vector relu(const Vector& x);
};

#endif // NEURAL_NETWORK_H
