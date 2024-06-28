#include "neural_network.h"
#include <Eigen/Dense>
#include <algorithm>

NeuralNetwork::NeuralNetwork(const std::vector<Layer>& layers) : layers_(layers) {}

Vector NeuralNetwork::forward(const Vector& input) {
    Vector x = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        x = layers_[i].weights * x + layers_[i].biases;
        if (i < layers_.size() - 1) {
            x = relu(x);
        }
    }
    return x;
}

Vector NeuralNetwork::relu(const Vector& x) {
    return x.unaryExpr([](float elem) { return std::max(0.0f, elem); });
}
