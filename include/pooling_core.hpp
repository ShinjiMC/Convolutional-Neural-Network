#pragma once
#include <vector>
#include "tensor.hpp"

enum PoolingType
{
    MAX,
    MIN,
    AVERAGE
};

class Pooling
{
private:
    int kernel_h, kernel_w;
    int stride;
    int padding;
    PoolingType type;

    std::vector<Tensor4D> masks;

    double pool_region(const std::vector<double> &region) const;

public:
    Pooling(int kernel_h, int kernel_w, int stride, int padding, PoolingType type);

    Tensor4D forward(const Tensor4D &input);
    Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output);
};
