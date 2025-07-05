#pragma once
#include <vector>

class Flatten
{
private:
    // Guarda la forma original para un posible reshape inverso
    int channels, height, width;

public:
    // Convierte un tensor [C][H][W] a vector 1D
    std::vector<double> forward(const std::vector<std::vector<std::vector<double>>> &input);

    // Reshape inverso (se puede llamar como backward)
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<double> &grad_output);

    // Reshape explícito (igual a backward, por legibilidad)
    std::vector<std::vector<std::vector<double>>> reshape(const std::vector<double> &input_flat);
};