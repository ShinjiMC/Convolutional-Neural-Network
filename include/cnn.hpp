#pragma once
#include <vector>
#include <memory>
#include <numeric>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "layer_cnn.hpp"
#include "tensor.hpp"
#include "mlp_core.hpp"
#include "tensor2d.hpp"

class NeuralNetwork
{
private:
    std::vector<std::unique_ptr<LayerCNN>> layers;
    Mlp mlp;
    Tensor2D output_2d;
    std::vector<Tensor4D> cached_4d_outputs; // para backward
    std::vector<Tensor2D> cached_2d_outputs; // para backward
    std::vector<std::vector<double>> activations;

public:
    NeuralNetwork() {}

    void set_mlp(const Mlp &m)
    {
        mlp = m;
    }
    void add_layer(std::unique_ptr<LayerCNN> layer)
    {
        layers.push_back(std::move(layer));
    }
    void forward(const Tensor4D &input)
    {
        Tensor4D current_4d = input;
        cached_4d_outputs.clear();
        cached_2d_outputs.clear();

        for (const auto &layer : layers)
        {
            if (layer->is_2d_output())
            {
                layer->forward(current_4d);
                cached_4d_outputs.push_back(current_4d);
                cached_2d_outputs.push_back(layer->output_2d());
            }
            else
            {
                layer->forward(current_4d);
                current_4d = layer->output_4d();
                cached_4d_outputs.push_back(current_4d);
            }
        }
        const Tensor2D &flattened = layers.empty() ? throw std::runtime_error("Empty CNN before MLP") : layers.back()->output_2d();
        mlp.forward(flattened.row(0), activations, false);
        output_2d = mlp.get_output();
    }
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
    void update_weights(double lr)
    {
        for (auto &layer : layers)
            layer->update_weights(lr);
    }

    void backward(const std::vector<double> &target)
    {
        // Paso 1: backward del MLP
        mlp.backward(cached_2d_outputs.back().row(0), activations, target);
        auto grad = mlp.get_grad();
        Tensor2D grad_output_2d(1, grad.size());
        grad_output_2d.set_row(0, grad);

        // Paso 2: backward por capas CNN en reversa
        Tensor4D grad_output_4d;
        bool switched_to_4d = false;

        for (int i = (int)layers.size() - 1; i >= 0; --i)
        {
            if (!switched_to_4d)
            {
                if (layers[i]->is_2d_output())
                {
                    // SOLO si la capa es 2D y tiene backward 2D (ej. FlattenLayer)
                    grad_output_4d = layers[i]->backward_from_2d(grad_output_2d);
                    switched_to_4d = true;
                }
                else
                {
                    throw std::runtime_error("Expected flatten as last 2D layer before MLP");
                }
            }
            else
                grad_output_4d = layers[i]->backward(grad_output_4d);
        }
    }
    void train(const Tensor4D &images, const std::vector<int> &labels, double &avg_loss)
    {
        size_t n = images.batch_size(); // imágenes en formato [N][C][H][W]
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        std::vector<double> target(mlp.get_n_outputs());
        double total_loss = 0.0;
        double penalty = 0.0;
        double loss_n = 0.0;
        for (size_t k = 0; k < n; ++k)
        {
            size_t i = indices[k];
            Tensor4D input(1, images.channels(), images.height(), images.width());
            for (int c = 0; c < images.channels(); ++c)
                for (int h = 0; h < images.height(); ++h)
                    for (int w = 0; w < images.width(); ++w)
                        input(0, c, h, w) = images(i, c, h, w);

            // Paso 2: one-hot encode
            one_hot_encode(labels[i], target);
            forward(input);
            loss_n = cross_entropy_loss(activations.back(), target);
            total_loss += loss_n;
            backward(target);
            // update_weights(0.0001);
            std::cout << "\rbatch " << (k + 1) << "/" << n
                      << " loss: " << std::fixed << std::setprecision(6) << loss_n << std::flush;
        }
        std::cout << std::endl;

        penalty = mlp.compute_penalty();
        std::cout << "\npenalty: " << penalty << std::endl;
        avg_loss = (total_loss + penalty) / n;
    }
    void train(const Tensor4D &images, const std::vector<int> &labels, double &avg_loss, size_t batch_size)
    {
        size_t n = images.batch_size();
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        std::vector<double> target(mlp.get_n_outputs());
        double total_loss = 0.0;
        double penalty = 0.0;
        size_t total_batches = (n + batch_size - 1) / batch_size;

        for (size_t batch_idx = 0, start = 0; start < n; ++batch_idx, start += batch_size)
        {
            size_t end = std::min(start + batch_size, n);
            size_t current_batch_size = end - start;

            Tensor4D batch_input(current_batch_size, images.channels(), images.height(), images.width());

            for (size_t b = 0; b < current_batch_size; ++b)
            {
                size_t i = indices[start + b];
                for (int c = 0; c < images.channels(); ++c)
                    for (int h = 0; h < images.height(); ++h)
                        for (int w = 0; w < images.width(); ++w)
                            batch_input(b, c, h, w) = images(i, c, h, w);
            }

            double batch_loss = 0.0;

            for (size_t b = 0; b < current_batch_size; ++b)
            {
                size_t i = indices[start + b];
                Tensor4D sample = batch_input.batch(b);

                one_hot_encode(labels[i], target);

                forward(sample); // Guarda activations.back()
                double loss = cross_entropy_loss(activations.back(), target);
                batch_loss += loss;
                total_loss += loss;

                backward(target);
                // update_weights(0.0001);
            }

            batch_loss /= current_batch_size;

            // Imprimir progreso con pérdida por batch
            std::cout << "\rbatch " << (batch_idx + 1) << "/" << total_batches
                      << " loss: " << std::fixed << std::setprecision(6) << batch_loss << std::flush;
        }

        std::cout << std::endl;

        penalty = mlp.compute_penalty();
        avg_loss = (total_loss + penalty) / n;
    }

    void evaluate(const Tensor2D &images, const std::vector<int> &labels, double &accuracy)
    {
        int correct = 0;
        std::vector<std::vector<double>> activations;

        for (int i = 0; i < images.batch_size(); ++i)
        {
            const std::vector<double> &input = images.row(i);
            mlp.forward(input, activations, false);
            int pred = std::distance(activations.back().begin(),
                                     std::max_element(activations.back().begin(), activations.back().end()));
            if (pred == labels[i])
                ++correct;
        }
        accuracy = 100.0 * correct / images.batch_size();
    }
    void evaluate(const Tensor4D &images, const std::vector<int> &labels, double &accuracy)
    {
        int correct = 0;
        size_t n = images.batch_size();
        for (size_t i = 0; i < n; ++i)
        {
            // Extraer un solo ejemplo del batch
            Tensor4D input(1, images.channels(), images.height(), images.width());
            for (int c = 0; c < images.channels(); ++c)
                for (int h = 0; h < images.height(); ++h)
                    for (int w = 0; w < images.width(); ++w)
                        input(0, c, h, w) = images(i, c, h, w);
            forward(input);
            const std::vector<double> &output = output_2d.row(0);
            int pred = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            if (pred == labels[i])
                ++correct;
        }
        accuracy = 100.0 * correct / n;
    }

    void test(const Tensor2D &images, const std::vector<int> &labels, double &accuracy)
    {
        evaluate(images, labels, accuracy);
    }
    void test(const Tensor4D &images, const std::vector<int> &labels, double &accuracy)
    {
        evaluate(images, labels, accuracy);
    }
    void train_test(const Tensor4D &train_images, const std::vector<int> &train_labels,
                    const Tensor4D &test_images, const std::vector<int> &test_labels,
                    bool Test, const std::string &dataset_filename, int epochs = 1000)
    {
        std::string base_name = std::filesystem::path(dataset_filename).stem().string();
        std::filesystem::path output_dir = std::filesystem::path("output") / base_name;
        std::filesystem::create_directories(output_dir);
        std::ofstream log_file(output_dir / "log.txt");

        if (!log_file)
        {
            std::cerr << "Can't open log file\n";
            return;
        }

        double average_loss = 0, train_accuracy = 0, test_accuracy = 0;
        double best_test_accuracy = -1.0;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            auto train_start = std::chrono::high_resolution_clock::now();
            train(train_images, train_labels, average_loss, 32); // Puedes ajustar el batch_size aquí
            auto train_end = std::chrono::high_resolution_clock::now();
            double train_time = std::chrono::duration<double>(train_end - train_start).count();

            evaluate(train_images, train_labels, train_accuracy);

            double test_time = 0.0;
            if (Test)
            {
                auto test_start = std::chrono::high_resolution_clock::now();
                test(test_images, test_labels, test_accuracy);
                auto test_end = std::chrono::high_resolution_clock::now();
                test_time = std::chrono::duration<double>(test_end - test_start).count();
            }

            std::ostringstream log;
            log << "Epoch " << (epoch + 1)
                << ", Train Loss: " << average_loss
                << ", Train Acc: " << train_accuracy << "%"
                << ", Train Time: " << train_time << "s";

            if (Test)
            {
                log << ", Test Acc: " << test_accuracy << "%"
                    << ", Test Time: " << test_time << "s";

                if (test_accuracy > best_test_accuracy)
                {
                    best_test_accuracy = test_accuracy;
                    mlp.save_data((output_dir / "best_model.dat").string());
                }
            }

            std::cout << log.str() << std::endl;
            log_file << log.str() << std::endl;

            if ((epoch + 1) % 10 == 0)
            {
                std::string checkpoint = (output_dir / ("epoch_" + std::to_string(epoch + 1) + ".dat")).string();
                mlp.save_data(checkpoint);
                std::cout << "Model saved at epoch " << (epoch + 1) << ".\n";
            }

            if (average_loss < 1e-5)
            {
                std::cout << "Stopping: early stopping criterion met.\n";
                break;
            }
        }
    }
};
