#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "tensor2d.hpp"

class Layer
{
public:
    virtual ~Layer() = default;

    virtual void forward(const Tensor4D &input) {}
    virtual void forward(const Tensor2D &input) {}

    virtual const Tensor4D &output_4d() const { throw std::runtime_error("No 4D output"); }
    virtual const Tensor2D &output_2d() const { throw std::runtime_error("No 2D output"); }

    virtual bool is_2d_output() const = 0;
};