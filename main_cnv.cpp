#include "conv2d.hpp"
#include "pooling.hpp"
#include "flatten.hpp"
#include <iostream>
#include <vector>
#include <iomanip> // para std::setw

void print_tensor(const std::vector<std::vector<std::vector<double>>> &tensor, const std::string &title)
{
    std::cout << title << ":\n";
    for (int c = 0; c < tensor.size(); ++c)
    {
        std::cout << "Canal " << c << ":\n";
        for (const auto &row : tensor[c])
        {
            for (double val : row)
                std::cout << std::setw(6) << val << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void print_vector(const std::vector<double> &vec, const std::string &title)
{
    std::cout << title << ":\n";
    for (double val : vec)
        std::cout << std::setw(6) << val << " ";
    std::cout << "\n\n";
}

int main()
{
    const int height = 3, width = 3, channels = 3;
    std::vector<std::vector<std::vector<double>>> input(
        channels, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                input[c][i][j] = c * 10 + i * 3 + j + 1;

    print_tensor(input, "Imagen de entrada [3x3x3]");

    // Convolución: in_channels=3, out_channels=3, kernel=2x2, stride=1, padding=0
    Conv2D conv(3, 3, 2, 2, 1, 0, ActivationType::RELU);
    auto conv_output = conv.forward(input);
    print_tensor(conv_output, "Salida de la convolución");

    // Pooling: kernel=2x2, stride=1, padding=0, MAX
    Pooling pool(2, 2, 1, 0, PoolingType::MAX);
    auto pooled_output = pool.forward(conv_output);
    print_tensor(pooled_output, "Salida del pooling (MAX)");

    // Flatten
    Flatten flatten;
    auto flat_output = flatten.forward(pooled_output);
    print_vector(flat_output, "Salida del flatten");

    // ============================================
    // Simula gradiente de pérdida (como si viniera de una capa Dense)
    std::vector<double> grad_from_loss(flat_output.size(), 1.0); // dL/d(flat_output) = 1s
    print_vector(grad_from_loss, "Gradiente desde la pérdida (dL/dy)");

    // Flatten.backward
    auto grad_flatten = flatten.backward(grad_from_loss);
    print_tensor(grad_flatten, "Gradiente hacia pooling");

    // Pooling.backward
    auto grad_pool = pool.backward(conv_output, grad_flatten);
    print_tensor(grad_pool, "Gradiente hacia convolución");

    // Conv2D.backward
    auto grad_conv = conv.backward(grad_pool);
    print_tensor(grad_conv, "Gradiente hacia imagen de entrada");

    return 0;
}
