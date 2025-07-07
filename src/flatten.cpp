#include "flatten_core.hpp"
#include <stdexcept>
#include <iostream>
Tensor2D Flatten::forward(const Tensor4D &input)
{
    int N = input.batch_size();
    channels = input.channels();
    height = input.height();
    width = input.width();

    int feature_size = channels * height * width;
    // std::cout << "[Flatten forward] = [" << N << "x" << channels << "x" << height << "x" << width << "]"
    //           << " ==> [" << N << "x" << feature_size << "]\n";
    Tensor2D output(N, feature_size);
    for (int n = 0; n < N; ++n)
    {
        int idx = 0;
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    output(n, idx++) = input(n, c, i, j);
    }
    return output;
}
Tensor4D Flatten::backward(const Tensor2D &grad_output)
{
    int N = grad_output.batch_size();
    int expected_size = channels * height * width;

    if (grad_output.feature_size() != expected_size)
        throw std::runtime_error("Flatten::backward - input size mismatch.");
    // std::cout << "[Flatten backward] = [" << N << "x" << expected_size << "]"
    //           << " ==> [" << N << "x" << channels << "x" << height << "x" << width << "]\n";
    Tensor4D output(N, channels, height, width);

    for (int n = 0; n < N; ++n)
    {
        int idx = 0;
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    output(n, c, i, j) = grad_output(n, idx++);
    }

    return output;
}