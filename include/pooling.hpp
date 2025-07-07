#pragma once
#include <memory>
#include "tensor.hpp"
#include "layer_cnn.hpp"
#include "pooling_core.hpp"

class PoolingLayer : public LayerCNN
{
private:
    std::unique_ptr<Pooling> pool;
    Tensor4D input_cache;
    Tensor4D output;

public:
    PoolingLayer(int kernel_h, int kernel_w, int stride, int padding, PoolingType type)
    {
        pool = std::make_unique<Pooling>(kernel_h, kernel_w, stride, padding, type);
    }

    void forward(const Tensor4D &input) override
    {
        input_cache = input;
        output = pool->forward(input);
    }

    const Tensor4D &output_4d() const override
    {
        return output;
    }

    Tensor4D backward(const Tensor4D &grad_output) override
    {
        return pool->backward(input_cache, grad_output);
    }

    void update_weights(double /*lr*/) override {}

    bool is_2d_output() const override { return false; }

    Pooling *get_pooling() { return pool.get(); }
};
