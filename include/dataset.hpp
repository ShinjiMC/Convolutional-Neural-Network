#pragma once
#include <string>
#include <vector>

class Dataset
{
private:
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;

public:
    Dataset(const std::string &filename);
    Dataset() = default;
    const std::vector<std::vector<double>> &get_X() const;
    const std::vector<std::vector<double>> &get_y() const;
    const std::vector<int> &get_ys() const;
    void print_data(std::string name) const;
    void generate_mnist(const std::string &filename, int TRAIN_SAMPLES, int TEST_SAMPLES);
};