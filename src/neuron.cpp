#include "neuron.hpp"
#include <cmath>

void Neuron::save(std::ostream &out) const
{
    for (const auto &w : weights)
        out << w << " ";
    out << bias << "\n";
}

void Neuron::load(std::istream &in, int n_inputs)
{
    weights.resize(n_inputs);
    for (int i = 0; i < n_inputs; ++i)
        in >> weights[i];
    in >> bias;
}

void Neuron::update(std::shared_ptr<Optimizer> opt, double learning_rate,
                    const double *input, double delta,
                    int input_size, int neuron_index, int layer_index)
{
    int global_id = layer_index * 100000 + neuron_index;
    opt->update(learning_rate, weights, bias,
                input, delta, input_size, global_id);
}

double Neuron::forward(const std::vector<double> &input) const
{
    double sum = bias;
    for (size_t i = 0; i < input.size(); ++i)
        sum += input[i] * weights[i];
    return sum;
}