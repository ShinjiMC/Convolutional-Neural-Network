#include "pooling.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

Pooling::Pooling(int kernel_h, int kernel_w, int stride, int padding, PoolingType type)
    : kernel_h(kernel_h), kernel_w(kernel_w), stride(stride), padding(padding), type(type) {}

double Pooling::pool_region(const std::vector<std::vector<double>> &region) const
{
    std::vector<double> flat;
    for (const auto &row : region)
        flat.insert(flat.end(), row.begin(), row.end());

    switch (type)
    {
    case PoolingType::MAX:
        return *std::max_element(flat.begin(), flat.end());
    case PoolingType::MIN:
        return *std::min_element(flat.begin(), flat.end());
    case PoolingType::AVERAGE:
        return std::accumulate(flat.begin(), flat.end(), 0.0) / flat.size();
    default:
        throw std::runtime_error("Unknown pooling type.");
    }
}

std::vector<std::vector<std::vector<double>>> Pooling::forward(
    const std::vector<std::vector<std::vector<double>>> &input)
{
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(
        channels, std::vector<std::vector<double>>(out_h, std::vector<double>(out_w, 0.0)));

    mask.clear();
    mask.resize(channels, std::vector<std::vector<std::vector<bool>>>(
                              out_h, std::vector<std::vector<bool>>(out_w,
                                                                    std::vector<bool>(kernel_h * kernel_w, false))));

    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < out_h; ++i)
        {
            for (int j = 0; j < out_w; ++j)
            {
                std::vector<std::vector<double>> region(kernel_h, std::vector<double>(kernel_w, 0.0));
                std::vector<std::pair<int, int>> coords;
                int idx = 0;

                for (int ki = 0; ki < kernel_h; ++ki)
                {
                    for (int kj = 0; kj < kernel_w; ++kj, ++idx)
                    {
                        int row = i * stride + ki - padding;
                        int col = j * stride + kj - padding;

                        if (row >= 0 && row < height && col >= 0 && col < width)
                            region[ki][kj] = input[c][row][col];
                        else
                            region[ki][kj] = 0.0; // padding explícito
                    }
                }

                double val = pool_region(region);
                output[c][i][j] = val;

                if (type == MAX || type == MIN)
                {
                    double target = val;
                    idx = 0;
                    for (int ki = 0; ki < kernel_h; ++ki)
                        for (int kj = 0; kj < kernel_w; ++kj, ++idx)
                            if (region[ki][kj] == target)
                                mask[c][i][j][idx] = true; // marca posición responsable
                }
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<double>>> Pooling::backward(
    const std::vector<std::vector<std::vector<double>>> &input,
    const std::vector<std::vector<std::vector<double>>> &grad_output)
{
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    std::vector<std::vector<std::vector<double>>> grad_input(
        channels, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));

    int out_h = grad_output[0].size();
    int out_w = grad_output[0][0].size();

    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < out_h; ++i)
        {
            for (int j = 0; j < out_w; ++j)
            {
                double grad = grad_output[c][i][j];
                int idx = 0;

                for (int ki = 0; ki < kernel_h; ++ki)
                {
                    for (int kj = 0; kj < kernel_w; ++kj, ++idx)
                    {
                        int row = i * stride + ki - padding;
                        int col = j * stride + kj - padding;

                        if (row >= 0 && row < height && col >= 0 && col < width)
                        {
                            if (type == AVERAGE)
                            {
                                grad_input[c][row][col] += grad / (kernel_h * kernel_w);
                            }
                            else if (type == MAX || type == MIN)
                            {
                                if (mask[c][i][j][idx])
                                    grad_input[c][row][col] += grad;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}