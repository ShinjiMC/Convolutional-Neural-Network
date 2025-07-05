#pragma once
#include <vector>
#include "tensor.hpp"

class Flatten
{
private:
    int channels, height, width;

public:
    // Convierte [N][C][H][W] → [N][C*H*W]
    std::vector<std::vector<double>> forward(const Tensor4D &input);

    // Convierte [N][C*H*W] → [N][C][H][W]
    Tensor4D backward(const std::vector<std::vector<double>> &grad_output);

    // Getter de dimensiones (para capas siguientes)
    int get_flattened_size() const { return channels * height * width; }
};