#pragma once
#include <random>
#include <omp.h>
#include "optimizer.hpp"
#include "activation.hpp"
#include <memory>
class Layer
{
private:
    int input_size;
    int output_size;
    ActivationType activation;

    std::vector<std::vector<double>> weights; // [output_size][input_size]
    std::vector<double> biases;               // [output_size]

public:
    Layer(int in_size, int out_size, ActivationType act)
        : input_size(in_size), output_size(out_size), activation(act)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<> dis(-limit, limit);
        weights.resize(output_size, std::vector<double>(input_size));
        biases.resize(output_size, 0.0);

        for (int i = 0; i < output_size; ++i)
            for (int j = 0; j < input_size; ++j)
                weights[i][j] = dis(gen);
    }
    Layer(int in_size, int out_size, ActivationType act, bool true_random)
        : input_size(in_size), output_size(out_size), activation(act)
    {
        weights.resize(output_size, std::vector<double>(input_size));
        biases.resize(output_size, 0.0);
    }
    void linear_forward(const std::vector<double> &input, std::vector<double> &output) const;
    void apply_activation(std::vector<double> &output) const;
    void apply_update(std::shared_ptr<Optimizer> optimizer,
                      const std::vector<double> &delta,
                      const std::vector<double> &input,
                      double learning_rate, int layer_index);
    // gets
    int get_input_size() { return input_size; }
    int get_output_size() { return output_size; }
    const int get_input_size() const { return input_size; }
    const int get_output_size() const { return output_size; }
    const int get_inputss() const { return input_size; }
    const int get_outputss() const { return output_size; }
    ActivationType get_activation() const { return activation; }
    void save(std::ostream &out, const int i) const;
    void load(std::istream &in);
    double compute_penalty() const;
    double get_weight(int output_idx, int input_idx) const
    {
        return weights[output_idx][input_idx];
    }
    const std::vector<std::vector<double>> &get_weights() const
    {
        return weights;
    }
};