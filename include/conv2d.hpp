#pragma once
#include <memory>
#include "tensor.hpp"
#include "layer_cnn.hpp"
#include "conv2d_core.hpp"
class Conv2DLayer : public LayerCNN
{
private:
    std::unique_ptr<Conv2D> conv;
    Tensor4D output;

public:
    Conv2DLayer(int in_channels, int out_channels, int kernel_h, int kernel_w,
                int stride = 1, int padding = 0, ActivationType activation = RELU)
    {
        conv = std::make_unique<Conv2D>(in_channels, out_channels, kernel_h, kernel_w, stride, padding, activation);
    }

    void forward(const Tensor4D &input) override
    {
        output = conv->forward(input);
    }

    const Tensor4D &output_4d() const override
    {
        return output;
    }

    Tensor4D backward(const Tensor4D &grad_output) override
    {
        return conv->backward(grad_output);
    }

    void update_weights(double lr) override
    {
        conv->update_weights(lr);
    }

    bool is_2d_output() const override
    {
        return false;
    }

    Conv2D *get_conv() { return conv.get(); }
};
