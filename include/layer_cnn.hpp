#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "tensor2d.hpp"

class LayerCNN
{
public:
    virtual ~LayerCNN() = default;

    virtual void forward(const Tensor4D &input) {}
    virtual void forward(const Tensor2D &input) {}

    virtual const Tensor4D &output_4d() const { throw std::runtime_error("No 4D output"); }
    virtual const Tensor2D &output_2d() const { throw std::runtime_error("No 2D output"); }

    virtual Tensor4D backward(const Tensor4D &grad_output) { throw std::runtime_error("No 4D backward"); }
    virtual Tensor2D backward(const Tensor2D &grad_output) { throw std::runtime_error("No 2D backward"); }
    virtual Tensor4D backward_from_2d(const Tensor2D &grad_output) { throw std::runtime_error("No 2D backward"); }

    virtual void update_weights(double lr) {}

    virtual bool is_2d_output() const = 0;
};
