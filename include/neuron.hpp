#pragma once
#include <vector>
#include <functional>
#include <fstream>
#include <stdexcept>
#include "optimizer.hpp"
#include <memory>

class Neuron
{
private:
    std::vector<double> weights;
    double bias;

public:
    Neuron() = default;

    template <typename RNG, typename Dist>
    Neuron(int n_inputs, RNG &gen, Dist &dis)
        : weights(n_inputs)
    {
        for (int i = 0; i < n_inputs; ++i)
            weights[i] = dis(gen);

        bias = 0.0;
    }
    std::vector<double> &get_weights() { return weights; }
    double &get_bias() { return bias; }
    const std::vector<double> &get_weightss() const { return weights; }
    const double &get_biass() const { return bias; }
    void save(std::ostream &out) const;
    void load(std::istream &in, int n_inputs);
    void update(std::shared_ptr<Optimizer> optimizer, double learning_rate,
                const double *input, double delta,
                int input_size, int neuron_index, int layer_index);
    void compute_penalty(double &penalty) const
    {
        for (const auto &weight : weights)
            penalty += weight * weight;
    }
    double forward(const std::vector<double> &input) const;
};