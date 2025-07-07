#pragma once
#include <vector>
#include "tensor.hpp"
#include "activation.hpp"

class Conv2D
{
private:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride, padding;
    ActivationType activation;

    Tensor4D filters; // [out_channels][in_channels][kH][kW]
    std::vector<double> biases;

    Tensor4D d_filters; // Gradientes
    std::vector<double> d_biases;

    Tensor4D last_input;
    Tensor4D pre_activations;
    Tensor4D output;

    void initialize_filters();

public:
    Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int stride = 1, int padding = 0, ActivationType activation = RELU);

    Tensor4D forward(const Tensor4D &batch_input);
    Tensor4D backward(const Tensor4D &grad_output);
    void update_weights(double lr);
};
