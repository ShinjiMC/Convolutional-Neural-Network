#pragma once
#include "layer.hpp"
#include <ctime>
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include "dropout.hpp"
#include "optimizer.hpp"
#include "regularizer.hpp"
#include "tensor2d.hpp"

class Mlp
{
private:
    int n_inputs;
    int n_outputs;
    double learning_rate;
    std::vector<Layer> layers;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::shared_ptr<Regularizer> regularizer = nullptr;
    std::shared_ptr<DropoutController> dropout = nullptr;
    Tensor2D output_tensor;
    std::vector<double> grad;

public:
    Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
        double lr, std::vector<ActivationType> activation_types,
        optimizer_type opt = optimizer_type::SGD,
        bool regularizer = false, bool dropout = false);
    Mlp() = default;
    Tensor2D get_output() const
    {
        return output_tensor;
    }
    const int get_n_inputs() const
    {
        return n_inputs;
    }
    const int get_n_outputs() const
    {
        return n_outputs;
    }
    const std::shared_ptr<Regularizer> get_regularizer() const
    {
        return regularizer;
    }
    double compute_penalty() const
    {
        if (regularizer)
            return regularizer->compute_penalty(layers);
        return 0.0;
    }

    std::vector<double> get_grad()
    {
        return grad;
    }

    void forward(const std::vector<double> &input,
                 std::vector<std::vector<double>> &activations,
                 bool train);
    void backward(const std::vector<double> &input,
                  const std::vector<std::vector<double>> &activations,
                  const std::vector<double> &expected);
    void one_hot_encode(int label, std::vector<double> &target)
    {
        std::fill(target.begin(), target.end(), 0.0);
        target[label] = 1.0;
    }
    double cross_entropy_loss(const std::vector<double> &predicted, const std::vector<double> &expected)
    {
        double loss = 0.0;
        const double epsilon = 1e-9;
        for (size_t i = 0; i < predicted.size(); ++i)
            loss -= expected[i] * log(predicted[i] + epsilon);
        return loss;
    }
    void train(const Tensor2D &images,
               const std::vector<int> &labels,
               double &avg_loss);
    void test(const Tensor2D &images,
              const std::vector<int> &labels,
              double &test_accuracy);
    void evaluate(const Tensor2D &images,
                  const std::vector<int> &labels,
                  double &train_accuracy);
    void train_test(const Tensor2D &train_images, const std::vector<int> &train_labels,
                    const Tensor2D &test_images, const std::vector<int> &test_labels,
                    bool Test, const std::string &dataset_filename, int epochs = 1000);
    void save_data(const std::string &filename) const;
    bool load_data(const std::string &filename);
};