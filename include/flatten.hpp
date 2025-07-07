#pragma once
#include <memory>
#include "tensor.hpp"
#include "tensor2d.hpp"
#include "layer_cnn.hpp"
#include "flatten_core.hpp"

class FlattenLayer : public LayerCNN
{
private:
    std::unique_ptr<Flatten> flatten;
    Tensor4D input_cache;
    Tensor2D output;

public:
    FlattenLayer()
    {
        flatten = std::make_unique<Flatten>();
    }

    void forward(const Tensor4D &input) override
    {
        input_cache = input;
        output = flatten->forward(input);
    }

    const Tensor2D &output_2d() const override
    {
        return output;
    }

    Tensor4D backward_from_2d(const Tensor2D &grad_output) override
    {
        return flatten->backward(grad_output);
    }

    void update_weights(double /*lr*/) override {}

    bool is_2d_output() const override { return true; }

    Flatten *get_flatten() { return flatten.get(); }
};
