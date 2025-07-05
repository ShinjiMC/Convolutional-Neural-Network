#include "conv2d.hpp"
#include <cmath>
#include <iostream>

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
               int stride, int padding, ActivationType activation_)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_h(kernel_h), kernel_w(kernel_w),
      stride(stride), padding(padding), activation(activation_)
{
    initialize_filters();
}

void Conv2D::initialize_filters()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    int fan_in = in_channels * kernel_h * kernel_w;
    int fan_out = out_channels * kernel_h * kernel_w;
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<double> dis(-limit, limit);

    filters.resize(out_channels, std::vector<std::vector<std::vector<double>>>(
                                     in_channels, std::vector<std::vector<double>>(
                                                      kernel_h, std::vector<double>(kernel_w))));
    biases.resize(out_channels, 0.0);
    for (int oc = 0; oc < out_channels; ++oc)
        for (int ic = 0; ic < in_channels; ++ic)
            for (int i = 0; i < kernel_h; ++i)
                for (int j = 0; j < kernel_w; ++j)
                    filters[oc][ic][i][j] = dis(gen);
}

std::vector<std::vector<std::vector<double>>> Conv2D::pad_input(
    const std::vector<std::vector<std::vector<double>>> &input)
{
    int h = input[0].size();
    int w = input[0][0].size();
    int padded_h = h + 2 * padding;
    int padded_w = w + 2 * padding;

    std::vector<std::vector<std::vector<double>>> padded_input(in_channels,
                                                               std::vector<std::vector<double>>(padded_h, std::vector<double>(padded_w, 0.0)));
    for (int c = 0; c < in_channels; ++c)
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                padded_input[c][i + padding][j + padding] = input[c][i][j];
    return padded_input;
}

std::vector<std::vector<std::vector<double>>> Conv2D::forward(
    const std::vector<std::vector<std::vector<double>>> &input)
{
    last_input = input;
    auto padded = (padding > 0) ? pad_input(input) : input;
    int h = padded[0].size();
    int w = padded[0][0].size();
    int out_h = (h - kernel_h) / stride + 1;
    int out_w = (w - kernel_w) / stride + 1;
    pre_activations.resize(out_channels,
                           std::vector<std::vector<double>>(out_h, std::vector<double>(out_w)));
    std::vector<std::vector<std::vector<double>>> output(out_channels,
                                                         std::vector<std::vector<double>>(out_h, std::vector<double>(out_w, 0.0)));
    for (int oc = 0; oc < out_channels; ++oc)
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
            {
                double sum = biases[oc];
                for (int ic = 0; ic < in_channels; ++ic)
                    for (int ki = 0; ki < kernel_h; ++ki)
                        for (int kj = 0; kj < kernel_w; ++kj)
                        {
                            int xi = i * stride + ki;
                            int xj = j * stride + kj;
                            sum += padded[ic][xi][xj] * filters[oc][ic][ki][kj];
                        }
                pre_activations[oc][i][j] = sum;
                if (activation == RELU)
                    sum = relu(sum);
                else if (activation == SIGMOID)
                    sum = sigmoid(sum);
                else if (activation == TANH)
                    sum = tanh_fn(sum);
                output[oc][i][j] = sum;
            }
    return output;
}

std::vector<std::vector<std::vector<double>>> Conv2D::backward(
    const std::vector<std::vector<std::vector<double>>> &grad_output)
{
    auto input = (padding > 0) ? pad_input(last_input) : last_input;

    int in_h = input[0].size();
    int in_w = input[0][0].size();
    int out_h = grad_output[0].size();
    int out_w = grad_output[0][0].size();

    d_filters.assign(out_channels,
                     std::vector<std::vector<std::vector<double>>>(in_channels,
                                                                   std::vector<std::vector<double>>(kernel_h,
                                                                                                    std::vector<double>(kernel_w, 0.0))));
    d_biases.assign(out_channels, 0.0);

    std::vector<std::vector<std::vector<double>>> grad_input(in_channels,
                                                             std::vector<std::vector<double>>(in_h, std::vector<double>(in_w, 0.0)));

    for (int oc = 0; oc < out_channels; ++oc)
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
            {
                double grad = grad_output[oc][i][j];
                double z = pre_activations[oc][i][j];

                if (activation == RELU)
                    grad *= relu_derivative(z);
                else if (activation == SIGMOID)
                    grad *= sigmoid_derivative(z);
                else if (activation == TANH)
                    grad *= tanh_derivative(z);

                d_biases[oc] += grad;

                for (int ic = 0; ic < in_channels; ++ic)
                    for (int ki = 0; ki < kernel_h; ++ki)
                        for (int kj = 0; kj < kernel_w; ++kj)
                        {
                            int xi = i * stride + ki;
                            int xj = j * stride + kj;

                            d_filters[oc][ic][ki][kj] += input[ic][xi][xj] * grad;
                            grad_input[ic][xi][xj] += filters[oc][ic][ki][kj] * grad;
                        }
            }

    if (padding > 0)
    {
        std::vector<std::vector<std::vector<double>>> unpadded(in_channels,
                                                               std::vector<std::vector<double>>(in_h - 2 * padding,
                                                                                                std::vector<double>(in_w - 2 * padding)));
        for (int c = 0; c < in_channels; ++c)
            for (int i = 0; i < in_h - 2 * padding; ++i)
                for (int j = 0; j < in_w - 2 * padding; ++j)
                    unpadded[c][i][j] = grad_input[c][i + padding][j + padding];
        return unpadded;
    }
    return grad_input;
}

void Conv2D::update_weights(double lr)
{
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int ic = 0; ic < in_channels; ++ic)
            for (int i = 0; i < kernel_h; ++i)
                for (int j = 0; j < kernel_w; ++j)
                    filters[oc][ic][i][j] -= lr * d_filters[oc][ic][i][j];
        biases[oc] -= lr * d_biases[oc];
    }
}