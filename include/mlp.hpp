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
    std::vector<Tensor2D> reusable_activations;
    std::vector<Tensor2D> reusable_deltas;

public:
    Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
        double lr, std::vector<ActivationType> activation_types,
        optimizer_type opt = optimizer_type::SGD,
        bool regularizer = false, bool dropout = false);
    Mlp() = default;
    void forward_batch(const Tensor2D &input_batch,
                       bool train);
    void backward_batch(const Tensor2D &input_batch,
                        const std::vector<int> &labels);
    void one_hot_encode_ptr(int label, double *target, int size)
    {
        std::fill(target, target + size, 0.0);
        target[label] = 1.0;
    }
    double cross_entropy_loss_ptr(const double *predicted, const double *expected, int size)
    {
        double loss = 0.0;
        const double epsilon = 1e-9;
        for (int i = 0; i < size; ++i)
            loss -= expected[i] * std::log(predicted[i] + epsilon);
        return loss;
    }
    void train_batch(const Tensor2D &input_batch,
                     const std::vector<int> &labels,
                     double &avg_loss);
    void train_with_batches(const Tensor2D &images,
                            const std::vector<int> &labels,
                            double &avg_loss,
                            int batch_size);
    void test(const Tensor2D &images,
              const std::vector<int> &labels,
              double &test_accuracy);
    void train_test(const Tensor2D &train_images, const std::vector<int> &train_labels,
                    const Tensor2D &test_images, const std::vector<int> &test_labels,
                    bool Test, const std::string &dataset_filename, int epochs = 1000);
    void save_data(const std::string &filename) const;
    bool load_data(const std::string &filename);
    void evaluate(const Tensor2D &images,
                  const std::vector<int> &labels,
                  double &train_accuracy);
};