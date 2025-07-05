#include "conv2d.hpp"
#include "pooling.hpp"
#include "flatten.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <iomanip> // para std::setw

void print_tensor4d(const Tensor4D &tensor, const std::string &title)
{
    std::cout << title << ":\n";
    for (int n = 0; n < tensor.batch_size(); ++n)
    {
        std::cout << "Ejemplo " << n << ":\n";
        for (int c = 0; c < tensor.channels(); ++c)
        {
            std::cout << "Canal " << c << ":\n";
            for (int i = 0; i < tensor.height(); ++i)
            {
                for (int j = 0; j < tensor.width(); ++j)
                    std::cout << std::setw(6) << tensor(n, c, i, j) << " ";
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
}

void print_vector_2d(const std::vector<std::vector<double>> &batch, const std::string &title)
{
    std::cout << title << ":\n";
    for (int n = 0; n < batch.size(); ++n)
    {
        std::cout << "Ejemplo " << n << ": ";
        for (double val : batch[n])
            std::cout << std::setw(6) << val << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main()
{
    const int N = 1; // batch size
    const int C = 3, H = 3, W = 3;
    Tensor4D input(N, C, H, W);

    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                input(0, c, i, j) = c * 10 + i * 3 + j + 1;

    print_tensor4d(input, "Imagen de entrada [N=1, C=3, H=3, W=3]");

    // Convolución
    Conv2D conv(C, 3, 2, 2, 1, 0, ActivationType::RELU);
    Tensor4D conv_output = conv.forward(input);
    print_tensor4d(conv_output, "Salida de la convolución");

    // Pooling
    Pooling pool(2, 2, 1, 0, PoolingType::MAX);
    Tensor4D pooled_output = pool.forward(conv_output);
    print_tensor4d(pooled_output, "Salida del pooling (MAX)");

    // Flatten
    Flatten flatten;
    auto flat_output = flatten.forward(pooled_output);
    print_vector_2d(flat_output, "Salida del flatten");

    // Gradiente simulado desde capa densa
    std::vector<std::vector<double>> grad_from_loss(flat_output.size(),
                                                    std::vector<double>(flat_output[0].size(), 1.0));
    print_vector_2d(grad_from_loss, "Gradiente desde la pérdida (dL/dy)");

    // Flatten.backward
    Tensor4D grad_flatten = flatten.backward(grad_from_loss);
    print_tensor4d(grad_flatten, "Gradiente hacia pooling");

    // Pooling.backward
    Tensor4D grad_pool = pool.backward(conv_output, grad_flatten);
    print_tensor4d(grad_pool, "Gradiente hacia convolución");

    // Conv2D.backward
    Tensor4D grad_conv = conv.backward(grad_pool);
    print_tensor4d(grad_conv, "Gradiente hacia imagen de entrada");

    return 0;
}
