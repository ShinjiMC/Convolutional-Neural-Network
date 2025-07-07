#pragma once
#include "tensor.hpp"
#include "tensor2d.hpp"

class Flatten
{
private:
    int channels, height, width;

public:
    Tensor2D forward(const Tensor4D &input);
    Tensor4D backward(const Tensor2D &grad_output);
    int get_flattened_size() const { return channels * height * width; }
};
