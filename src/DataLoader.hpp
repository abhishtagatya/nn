#ifndef DATALOADER_HPP
#define DATALOADER_HPP
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

typedef std::vector<std::vector<float>> csvFile;

static inline csvFile loadCSV(const std::string& path, const float norm_value = 1.0f) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + path);

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell) / norm_value);
        }

        data.push_back(std::move(row));
    }

    return data;
}

static inline void flatten(csvFile& data) {
    for (auto& row : data) {
        for (auto& cell : row) {
            cell /= 255.0;
        }
    }
}

static inline void dumpCSV(const std::string& path, const std::vector<int>& preds) {
    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Could not open file for writing: " + path);

    for (int p : preds) {
        file << p << "\n";
    }
}

#endif //DATALOADER_HPP
