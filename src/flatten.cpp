#include "flatten.hpp"
#include <stdexcept>

std::vector<std::vector<double>> Flatten::forward(const Tensor4D &input)
{
    int N = input.batch_size();
    channels = input.channels();
    height = input.height();
    width = input.width();
    int size = channels * height * width;
    std::vector<std::vector<double>> output(N, std::vector<double>(size));
    for (int n = 0; n < N; ++n)
    {
        int idx = 0;
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    output[n][idx++] = input(n, c, i, j);
    }
    return output;
}

Tensor4D Flatten::backward(const std::vector<std::vector<double>> &grad_output)
{
    int N = grad_output.size();
    int expected_size = channels * height * width;
    Tensor4D output(N, channels, height, width);
    for (int n = 0; n < N; ++n)
    {
        if ((int)grad_output[n].size() != expected_size)
            throw std::runtime_error("Flatten::backward - input size mismatch.");
        int idx = 0;
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    output(n, c, i, j) = grad_output[n][idx++];
    }
    return output;
}