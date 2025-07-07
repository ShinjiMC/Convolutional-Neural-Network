#include "conv2d_core.hpp"
#include <cmath>
#include <iostream>
#include <random>

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
    double limit = std::sqrt(6.0 / (in_channels * kernel_h * kernel_w + out_channels * kernel_h * kernel_w));
    std::uniform_real_distribution<double> dis(-limit, limit);
    filters = Tensor4D(out_channels, in_channels, kernel_h, kernel_w);
    for (int oc = 0; oc < out_channels; ++oc)
        for (int ic = 0; ic < in_channels; ++ic)
            for (int i = 0; i < kernel_h; ++i)
                for (int j = 0; j < kernel_w; ++j)
                    filters(oc, ic, i, j) = dis(gen);
    biases.assign(out_channels, 0.0);
}

Tensor4D Conv2D::forward(const Tensor4D &batch_input)
{
    int N = batch_input.batch_size();
    int H = batch_input.height();
    int W = batch_input.width();
    int out_h = (H + 2 * padding - kernel_h) / stride + 1;
    int out_w = (W + 2 * padding - kernel_w) / stride + 1;
    // std::cout << "[Conv2D forward] = [" << N << "x" << in_channels << "x" << H << "x" << W << "]"
    //           << " ==> [" << N << "x" << out_channels << "x" << out_h << "x" << out_w << "]" << std::endl;
    Tensor4D output(N, out_channels, out_h, out_w);
    last_input = batch_input;
    pre_activations = Tensor4D(N, out_channels, out_h, out_w);
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j)
                {
                    double sum = biases[oc];
                    for (int ic = 0; ic < in_channels; ++ic)
                        for (int ki = 0; ki < kernel_h; ++ki)
                            for (int kj = 0; kj < kernel_w; ++kj)
                            {
                                int xi = i * stride + ki - padding;
                                int xj = j * stride + kj - padding;
                                if (xi >= 0 && xi < H && xj >= 0 && xj < W)
                                    sum += batch_input(n, ic, xi, xj) * filters(oc, ic, ki, kj);
                            }
                    pre_activations(n, oc, i, j) = sum;
                    if (activation == RELU)
                        sum = relu(sum);
                    else if (activation == SIGMOID)
                        sum = sigmoid(sum);
                    else if (activation == TANH)
                        sum = tanh_fn(sum);
                    output(n, oc, i, j) = sum;
                }
    return output;
}

Tensor4D Conv2D::backward(const Tensor4D &grad_output)
{
    int N = grad_output.batch_size();
    int out_h = grad_output.height();
    int out_w = grad_output.width();
    int in_h = last_input.height();
    int in_w = last_input.width();

    Tensor4D grad_input(N, in_channels, in_h, in_w, 0.0);
    d_filters = Tensor4D(out_channels, in_channels, kernel_h, kernel_w, 0.0);
    d_biases.assign(out_channels, 0.0);
    // std::cout << "[Conv2D backward] grad_output = [" << N << "x" << out_channels << "x" << out_h << "x" << out_w << "]"
    //           << " ==> grad_input = [" << N << "x" << in_channels << "x" << in_h << "x" << in_w << "]" << std::endl;
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j)
                {
                    double grad = grad_output(n, oc, i, j);
                    double z = pre_activations(n, oc, i, j);
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
                                int xi = i * stride + ki - padding;
                                int xj = j * stride + kj - padding;
                                if (xi >= 0 && xi < in_h && xj >= 0 && xj < in_w)
                                {
                                    double input_val = last_input(n, ic, xi, xj);
                                    d_filters(oc, ic, ki, kj) += input_val * grad;

                                    // filtro reflejado (convolución transpuesta)
                                    int rki = kernel_h - 1 - ki;
                                    int rkj = kernel_w - 1 - kj;
                                    grad_input(n, ic, xi, xj) += filters(oc, ic, rki, rkj) * grad;
                                }
                            }
                }

    // 🔧 Normalización por batch
    double scale = 1.0 / N;
    for (int oc = 0; oc < out_channels; ++oc)
    {
        d_biases[oc] *= scale;
        for (int ic = 0; ic < in_channels; ++ic)
            for (int ki = 0; ki < kernel_h; ++ki)
                for (int kj = 0; kj < kernel_w; ++kj)
                    d_filters(oc, ic, ki, kj) *= scale;
    }

    return grad_input;
}

void Conv2D::update_weights(double lr)
{
    // double max_grad = 0.0;
    // for (int oc = 0; oc < out_channels; ++oc)
    //     for (int ic = 0; ic < in_channels; ++ic)
    //         for (int i = 0; i < kernel_h; ++i)
    //             for (int j = 0; j < kernel_w; ++j)
    //                 max_grad = std::max(max_grad, std::abs(d_filters(oc, ic, i, j)));

    // std::cout << "[Conv2D update_weights] max_grad = " << max_grad << std::endl;
    // std::cout << "[Conv2D update_weights] learning rate = " << lr << std::endl;
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int ic = 0; ic < in_channels; ++ic)
            for (int i = 0; i < kernel_h; ++i)
                for (int j = 0; j < kernel_w; ++j)
                    filters(oc, ic, i, j) -= lr * d_filters(oc, ic, i, j);
        biases[oc] -= lr * d_biases[oc];
    }
    d_filters.fill(0.0);
    std::fill(d_biases.begin(), d_biases.end(), 0.0);
}
