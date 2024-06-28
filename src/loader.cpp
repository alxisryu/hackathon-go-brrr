#include "loader.h"
#include <iostream>
#include <fstream>
#include <sstream>

void loadWeightsAndBiases(const std::string& filename, std::vector<Layer>& layers) {
    std::ifstream file(filename);
    std::string line;
    Layer* currentLayer = nullptr;

    while (std::getline(file, line)) {
        if (line.find("weight") != std::string::npos) {
            layers.emplace_back();
            currentLayer = &layers.back();
            std::getline(file, line);
            std::istringstream iss(line);
            std::vector<float> values;
            std::string value;
            while (std::getline(iss, value, ',')) {
                values.push_back(std::stof(value));
            }
            int rows = std::stoi(line.substr(2, 1)); // Assuming "fc1.weight" format
            int cols = values.size() / rows;
            currentLayer->weights = Matrix::Map(values.data(), rows, cols);
        } else if (line.find("bias") != std::string::npos) {
            std::getline(file, line);
            std::istringstream iss(line);
            std::vector<float> values;
            std::string value;
            while (std::getline(iss, value, ',')) {
                values.push_back(std::stof(value));
            }
            currentLayer->biases = Vector::Map(values.data(), values.size());
        }
    }
}
