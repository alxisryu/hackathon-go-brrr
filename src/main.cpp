#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <Eigen/Dense>
#include "loader.h"
#include "neural_network.h"

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

const std::string PREDICTIONS_FILENAME = "results.csv";

std::vector<Vector> loadTensors(const std::string& directory) {
    std::vector<Vector> tensors;
    for (const auto& entry : std::__fs::filesystem::directory_iterator(directory)) {
        std::ifstream file(entry.path());
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);
        std::vector<float> values;
        std::string value;
        while (std::getline(iss, value, ',')) {
            values.push_back(std::stof(value));
        }
        tensors.emplace_back(Vector::Map(values.data(), values.size()));
    }
    return tensors;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <relative_path_to_weights_and_biases.txt> <relative_path_to_input_tensor_directory>" << std::endl;
        return 1;
    }

    std::string weights_and_biases_path = argv[1];
    std::string tensor_dir = argv[2];

    std::vector<Layer> layers;
    loadWeightsAndBiases(weights_and_biases_path, layers);

    NeuralNetwork model(layers);

    std::vector<Vector> tensors = loadTensors(tensor_dir);

    std::vector<std::pair<int, char>> predictions;
    std::string lookup_table = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";

    for (size_t i = 0; i < tensors.size(); ++i) {
        Vector output = model.forward(tensors[i]);
        int maxIndex;
        output.maxCoeff(&maxIndex);
        predictions.emplace_back(i + 1, lookup_table[maxIndex]);
    }

    std::ofstream outputFile(PREDICTIONS_FILENAME);
    outputFile << "image number,label\n";
    for (const auto& p : predictions) {
        outputFile << p.first << "," << p.second << "\n";
    }

    return 0;
}
