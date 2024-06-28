#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <iomanip>

class NeuralNetwork {
public:
    NeuralNetwork() {
        // Initialize network architecture and allocate memory for weights and biases
        initialize_network();
    }

    void load_weights_and_biases(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open weights and biases file.\n";
            exit(1);
        }

        std::string line;
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            for (size_t i = 0; i < weights[layer].size(); ++i) {
                std::getline(file, line);
                std::istringstream iss(line);
                double value;
                while (iss >> value) {
                    weights[layer][i].push_back(value);
                }
            }
            std::getline(file, line);
            std::istringstream iss(line);
            double value;
            while (iss >> value) {
                biases[layer].push_back(value);
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> layer_input = input;

        for (size_t layer = 0; layer < weights.size(); ++layer) {
            std::vector<double> layer_output(weights[layer].size());
            for (size_t i = 0; i < weights[layer].size(); ++i) {
                layer_output[i] = biases[layer][i];
                for (size_t j = 0; j < weights[layer][i].size(); ++j) {
                    layer_output[i] += layer_input[j] * weights[layer][i][j];
                }
                if (layer != weights.size() - 1) { // Apply ReLU activation
                    layer_output[i] = std::max(0.0, layer_output[i]);
                }
            }
            layer_input = layer_output;
        }

        return softmax(layer_input);
    }

private:
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    void initialize_network() {
        // Define network architecture
        weights = {
            std::vector<std::vector<double>>(98, std::vector<double>(225)),
            std::vector<std::vector<double>>(65, std::vector<double>(98)),
            std::vector<std::vector<double>>(50, std::vector<double>(65)),
            std::vector<std::vector<double>>(30, std::vector<double>(50)),
            std::vector<std::vector<double>>(25, std::vector<double>(30)),
            std::vector<std::vector<double>>(40, std::vector<double>(25)),
            std::vector<std::vector<double>>(52, std::vector<double>(40))
        };
        biases = {
            std::vector<double>(98),
            std::vector<double>(65),
            std::vector<double>(50),
            std::vector<double>(30),
            std::vector<double>(25),
            std::vector<double>(40),
            std::vector<double>(52)
        };
    }

    std::vector<double> softmax(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        double sum = 0.0;
        for (double val : input) {
            sum += std::exp(val);
        }
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i]) / sum;
        }
        return output;
    }
};

std::vector<std::string> list_files(const std::string& directory) {
    // Implement listing files in a directory (platform-dependent, e.g., dirent.h for POSIX)
    std::vector<std::string> files;
    // Dummy implementation, replace with actual directory listing
    for (int i = 1; i <= 52; ++i) {
        std::ostringstream oss;
        oss << directory << "/" << std::setw(2) << std::setfill('0') << i << "out.txt";
        files.push_back(oss.str());
    }
    return files;
}

std::vector<double> load_tensor(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open tensor file " << filename << "\n";
        exit(1);
    }

    std::vector<double> tensor;
    std::string line;
    while (std::getline(file, line, ',')) {
        tensor.push_back(std::stod(line));
    }
    return tensor;
}

void write_predictions(const std::vector<std::pair<int, char>>& predictions, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open predictions file.\n";
        exit(1);
    }

    file << "image_number, guess\n";
    for (const auto& p : predictions) {
        file << p.first << ", " << p.second << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: speed_cpu <relative_path_to_weights_and_biases.txt> <relative_path_to_input_tensor_directory>\n";
        return 1;
    }

    std::string weights_and_biases = argv[1];
    std::string input_tensor_dir = argv[2];

    NeuralNetwork model;
    model.load_weights_and_biases(weights_and_biases);

    auto tensor_files = list_files(input_tensor_dir);

    std::vector<std::pair<int, char>> predictions;
    std::vector<char> lookup_table = {
        'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k',
        'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v',
        'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'
    };

    for (size_t i = 0; i < tensor_files.size(); ++i) {
        auto tensor = load_tensor(tensor_files[i]);
        auto output = model.forward(tensor);
        auto max_it = std::max_element(output.begin(), output.end());
        int predicted_label = std::distance(output.begin(), max_it);
        predictions.emplace_back(i + 1, lookup_table[predicted_label]);
    }

    write_predictions(predictions, "results.csv");

    return 0;
}