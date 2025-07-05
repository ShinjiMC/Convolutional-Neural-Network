#pragma once
#include <cmath>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <algorithm>

enum ActivationType
{
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

// --- Activations ---
inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

inline double relu(double x)
{
    return x > 0 ? x : 0;
}

inline double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

inline double tanh_fn(double x)
{
    return std::tanh(x);
}

inline double tanh_derivative(double x)
{
    double t = std::tanh(x);
    return 1.0 - t * t;
}

inline void softmax(const std::vector<double> &input, std::vector<double> &output)
{
    double max_val = *max_element(input.begin(), input.end());
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    for (double &val : output)
        val /= sum;
}

inline const std::unordered_map<ActivationType, std::string> activation_to_string = {
    {SIGMOID, "sigmoid"},
    {RELU, "relu"},
    {TANH, "tanh"},
    {SOFTMAX, "softmax"}};

inline const std::unordered_map<std::string, ActivationType> string_to_activation = {
    {"sigmoid", SIGMOID},
    {"relu", RELU},
    {"tanh", TANH},
    {"softmax", SOFTMAX}};

inline std::string to_string(ActivationType type)
{
    auto it = activation_to_string.find(type);
    return it != activation_to_string.end() ? it->second : "unknown";
}

inline ActivationType from_string(const std::string &name)
{
    auto it = string_to_activation.find(name);
    return it != string_to_activation.end() ? it->second : SIGMOID;
}
