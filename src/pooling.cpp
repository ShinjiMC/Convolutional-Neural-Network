#include "pooling_core.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>
Pooling::Pooling(int kernel_h, int kernel_w, int stride, int padding, PoolingType type)
    : kernel_h(kernel_h), kernel_w(kernel_w), stride(stride), padding(padding), type(type) {}

double Pooling::pool_region(const std::vector<double> &region) const
{
    switch (type)
    {
    case MAX:
        return *std::max_element(region.begin(), region.end());
    case MIN:
        return *std::min_element(region.begin(), region.end());
    case AVERAGE:
        return std::accumulate(region.begin(), region.end(), 0.0) / region.size();
    default:
        throw std::runtime_error("Unknown pooling type.");
    }
}

Tensor4D Pooling::forward(const Tensor4D &input)
{
    int N = input.batch_size();
    int C = input.channels();
    int H = input.height();
    int W = input.width();
    int out_h = (H + 2 * padding - kernel_h) / stride + 1;
    int out_w = (W + 2 * padding - kernel_w) / stride + 1;
    // std::cout << "[Pooling forward] = [" << N << "x" << C << "x" << H << "x" << W << "]"
    //           << " ==> [" << N << "x" << C << "x" << out_h << "x" << out_w << "]\n";
    Tensor4D output(N, C, out_h, out_w);
    masks.clear();
    if (type == MAX || type == MIN)
        masks.resize(N, Tensor4D(C, out_h, out_w, kernel_h * kernel_w));
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j)
                {
                    std::vector<double> region;
                    region.reserve(kernel_h * kernel_w);
                    for (int ki = 0; ki < kernel_h; ++ki)
                        for (int kj = 0; kj < kernel_w; ++kj)
                        {
                            int row = i * stride + ki - padding;
                            int col = j * stride + kj - padding;
                            double val = 0.0;
                            if (row >= 0 && row < H && col >= 0 && col < W)
                                val = input(n, c, row, col);
                            region.push_back(val);
                        }
                    double pooled_val = pool_region(region);
                    output(n, c, i, j) = pooled_val;
                    if (type == MAX || type == MIN)
                        for (int idx = 0; idx < region.size(); ++idx)
                            if (region[idx] == pooled_val)
                            {
                                masks[n](c, i, j, idx) = 1.0;
                                break; // solo el primero
                            }
                }
    return output;
}

Tensor4D Pooling::backward(const Tensor4D &input, const Tensor4D &grad_output)
{
    int N = input.batch_size();
    int C = input.channels();
    int H = input.height();
    int W = input.width();
    int out_h = grad_output.height();
    int out_w = grad_output.width();
    // std::cout << "[Pooling backward] [" << N << "x" << C << "x" << out_h << "x" << out_w << "]"
    //           << " ==> [" << N << "x" << C << "x" << H << "x" << W << "]\n";
    Tensor4D grad_input(N, C, H, W); // inicializado en 0
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j)
                {
                    double grad = grad_output(n, c, i, j);
                    for (int ki = 0; ki < kernel_h; ++ki)
                        for (int kj = 0; kj < kernel_w; ++kj)
                        {
                            int row = i * stride + ki - padding;
                            int col = j * stride + kj - padding;
                            if (row >= 0 && row < H && col >= 0 && col < W)
                            {
                                if (type == AVERAGE)
                                    grad_input(n, c, row, col) += grad / (kernel_h * kernel_w);
                                else if (type == MAX || type == MIN)
                                {
                                    int idx = ki * kernel_w + kj;
                                    if (masks[n](c, i, j, idx) == 1.0)
                                        grad_input(n, c, row, col) += grad;
                                }
                            }
                        }
                }
    return grad_input;
}