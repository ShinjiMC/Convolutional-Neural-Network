#include "flatten.hpp"
#include <stdexcept>

std::vector<double> Flatten::forward(const std::vector<std::vector<std::vector<double>>> &input)
{
    channels = input.size();
    height = input[0].size();
    width = input[0][0].size();

    std::vector<double> flat;
    flat.reserve(channels * height * width);

    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                flat.push_back(input[c][i][j]);

    return flat;
}

std::vector<std::vector<std::vector<double>>> Flatten::reshape(const std::vector<double> &input_flat)
{
    if ((int)input_flat.size() != channels * height * width)
        throw std::runtime_error("Input size does not match stored shape.");

    std::vector<std::vector<std::vector<double>>> output(
        channels, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    int idx = 0;
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                output[c][i][j] = input_flat[idx++];

    return output;
}

std::vector<std::vector<std::vector<double>>> Flatten::backward(const std::vector<double> &grad_output)
{
    return reshape(grad_output); // mismo comportamiento
}
