#pragma once
#include <vector>
#include "activation.hpp"
#include "tensor.hpp"

class Conv2D
{
private:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride, padding;
    ActivationType activation;

    Tensor4D filters; // [out_channels][in_channels][kH][kW]
    std::vector<double> biases;

    Tensor4D d_filters; // same as filters
    std::vector<double> d_biases;

    Tensor4D last_input;      // [batch][in_channels][H][W]
    Tensor4D pre_activations; // [batch][out_channels][H_out][W_out]
    void initialize_filters();

public:
    Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int stride = 1, int padding = 0, ActivationType activation = RELU);

    Tensor4D forward(const Tensor4D &batch_input);
    Tensor4D backward(const Tensor4D &grad_output);
    void update_weights(double lr);
};
