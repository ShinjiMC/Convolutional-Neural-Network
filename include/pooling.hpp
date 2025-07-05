#pragma once
#include <vector>
#include <string>

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

    std::vector<std::vector<std::vector<std::vector<bool>>>> mask;

    double pool_region(const std::vector<std::vector<double>> &region) const;

public:
    Pooling(int kernel_h, int kernel_w, int stride, int padding, PoolingType type);

    // Aplica pooling a una entrada [C][H][W]
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>> &input);

    // bvackward
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<std::vector<double>>> &input,
        const std::vector<std::vector<std::vector<double>>> &grad_output);
};