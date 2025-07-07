#include "layer.hpp"
void Layer::linear_forward(const std::vector<double> &input,
                           std::vector<double> &output) const
{
    output.resize(output_size);
#pragma omp parallel for
    for (int i = 0; i < output_size; ++i)
    {
        double sum = biases[i];
        for (int j = 0; j < input_size; ++j)
            sum += weights[i][j] * input[j];
        output[i] = sum;
    }
}
void Layer::apply_activation(std::vector<double> &output) const
{
    if (this->activation == SIGMOID)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)output.size(); ++i)
            output[i] = sigmoid(output[i]);
    }
    else if (this->activation == RELU)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)output.size(); ++i)
            output[i] = relu(output[i]);
    }
    else if (this->activation == TANH)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)output.size(); ++i)
            output[i] = tanh(output[i]);
    }
    else if (this->activation == SOFTMAX)
        softmax(output, output);
}

void Layer::save(std::ostream &out, const int i) const
{
    for (size_t j = 0; j < output_size; ++j)
    {
        out << i + 1 << " " << j + 1 << " ";
        for (int k = 0; k < input_size; ++k)
            out << weights[j][k] << " ";
        out << biases[j] << "\n";
    }
}

void Layer::load(std::istream &in)
{
    for (int j = 0; j < output_size; ++j)
    {
        int layer_idx, neuron_idx;
        in >> layer_idx >> neuron_idx;

        if (in.fail())
            throw std::runtime_error("Error al leer índice de neurona");

        for (int k = 0; k < input_size; ++k)
        {
            in >> weights[j][k];
            if (in.fail())
                throw std::runtime_error("Error al leer peso");
        }

        in >> biases[j];
        if (in.fail())
            throw std::runtime_error("Error al leer bias");
    }
}

void Layer::apply_update(std::shared_ptr<Optimizer> optimizer, const std::vector<double> &delta,
                         const std::vector<double> &input,
                         double learning_rate, int layer_index)
{
    for (int i = 0; i < output_size; ++i)
    {
        int global_id = layer_index * 100000 + i;
        optimizer->update(learning_rate, weights[i], biases[i],
                          input.data(), delta[i], input_size, global_id);
    }
}

double Layer::compute_penalty() const
{
    double sum = 0.0;

    if (weights.empty())
    {
        std::cout << "Vacio weights" << std::endl;
        return sum; // Evitar división por cero
    }
    if (weights[0].empty())
    {
        std::cout << "Vacio weights[0]" << std::endl;
        return sum; // Evitar división por cero
    }
    size_t rows = weights.size();
    size_t cols = weights[0].size();
#pragma omp parallel for reduction(+ : sum)
    for (size_t idx = 0; idx < rows * cols; ++idx)
    {
        size_t i = idx / cols;
        size_t j = idx % cols;
        sum += weights[i][j] * weights[i][j];
    }
    return sum;
}