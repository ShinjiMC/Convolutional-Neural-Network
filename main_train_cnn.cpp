#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include "mlp_core.hpp"
#include "dataset.hpp"
#include "config.hpp"
#include "cnn.hpp"
#include "conv2d.hpp"
#include "pooling.hpp"
#include "flatten.hpp"

std::string get_last_folder(const std::string &path)
{
    std::string cleaned_path = path;
    while (!cleaned_path.empty() && (cleaned_path.back() == '/' || cleaned_path.back() == '\\'))
        cleaned_path.pop_back();
    size_t slash = cleaned_path.find_last_of("/\\");
    if (slash == std::string::npos)
        return cleaned_path;
    return cleaned_path.substr(slash + 1);
}

Tensor2D to_tensor2d(const std::vector<std::vector<double>> &data)
{
    if (data.empty())
        return Tensor2D(0, 0);
    int batch_size = data.size();
    int n_features = data[0].size();
    Tensor2D tensor(batch_size, n_features);
    for (int i = 0; i < batch_size; ++i)
        tensor.set_row(i, std::move(data[i]));
    return tensor;
}

Tensor4D to_tensor4d(const Tensor2D &images_2d, int height, int width, int channels = 1)
{
    int N = images_2d.batch_size();
    Tensor4D images_4d(N, channels, height, width);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < height * width; ++j)
        {
            int h = j / width;
            int w = j % width;
            double pixel = images_2d(i, j);
            for (int c = 0; c < channels; ++c)
                images_4d(i, c, h, w) = pixel; // Replicamos el mismo valor en los 3 canales
        }
    }
    return images_4d;
}

int main(int argc, char *argv[])
{
    bool use_saved_model = false;
    bool epochs_train = false;
    int epochs = 0;
    std::string dataset_dir = "./database/MNIST/";
    int train_samples = 0;
    int test_samples = 0;
    bool generate_mnist = false;
    std::string save_path = "./output/MNIST/final.dat";
    std::string config_path = "config/mnist.txt";

    // Procesar argumentos
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--save_data" && i + 1 < argc)
        {
            save_path = argv[++i];
            use_saved_model = true;
        }
        else if (arg == "--epochs" && i + 1 < argc)
        {
            epochs = std::atoi(argv[++i]);
            epochs_train = true;
        }
        else if (arg == "--dataset" && i + 1 < argc)
        {
            dataset_dir = argv[++i];
            if (dataset_dir.back() != '/')
                dataset_dir += '/';
        }
        else if (arg == "--mnist" && i + 2 < argc)
        {
            train_samples = std::atoi(argv[++i]);
            test_samples = std::atoi(argv[++i]);
            generate_mnist = true;
        }
        else if (arg == "--config" && i + 1 < argc)
        {
            config_path = argv[++i];
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    if (generate_mnist)
    {
        Dataset m;
        std::cout << "Generating MNIST text data...\n";
        m.generate_mnist(dataset_dir, train_samples, test_samples);
    }

    std::string train_file = dataset_dir + "train.txt";
    std::string test_file = dataset_dir + "test.txt";

    Dataset train(train_file);
    std::vector<std::vector<double>> X_train = train.get_X();
    std::vector<int> y_train = train.get_ys();

    Dataset test(test_file);
    std::vector<std::vector<double>> X_test = test.get_X();
    std::vector<int> y_test = test.get_ys();

    train.print_data("TRAIN");
    test.print_data("TEST");

    NeuralNetwork net;
    int in_channels = 3; // Imagen RGB (simulada)
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    int pool_size = 2;
    int pool_stride = 2;
    int pool_padding = 0;

    // Conv2D: [3,28,28] → [16,28,28]
    net.add_layer(std::make_unique<Conv2DLayer>(
        in_channels, 16, kernel_size, kernel_size, stride, padding, RELU));

    // MaxPooling: [16,28,28] → [16,14,14]
    net.add_layer(std::make_unique<PoolingLayer>(
        pool_size, pool_size, pool_stride, pool_padding, PoolingType::MAX));

    // Conv2D: [16,14,14] → [4,14,14]
    net.add_layer(std::make_unique<Conv2DLayer>(
        16, 4, kernel_size, kernel_size, stride, padding, RELU));

    // MaxPooling: [4,14,14] → [4,7,7]
    net.add_layer(std::make_unique<PoolingLayer>(
        pool_size, pool_size, pool_stride, pool_padding, PoolingType::MAX));

    // Flatten: [4,7,7] → [1,196]
    net.add_layer(std::make_unique<FlattenLayer>());
    int n_inputs = X_train[0].size();
    Config cfg;
    if (!cfg.load_config(config_path, 196))
    {
        std::cerr << "Failed to load configuration.\n";
        return 1;
    }

    cfg.print_config();

    Mlp nn;

    if (use_saved_model)
    {
        if (!nn.load_data(save_path))
        {
            std::cerr << "Error loading saved model.\n";
            return 1;
        }
        std::cout << "Model loaded successfully.\n";
    }
    else
    {
        std::cout << "Initializing new neural network...\n";
        nn = Mlp(196, cfg.get_layer_sizes(),
                 cfg.get_layer_sizes().back(), cfg.get_learning_rate(),
                 cfg.get_activations(), cfg.get_optimizer(), true, true);
    }

    std::string dataset_name = get_last_folder(dataset_dir);
    save_path = "./output/" + dataset_name + "/final.dat";
    Tensor2D X_train_tensor = to_tensor2d(std::move(X_train));
    Tensor2D X_test_tensor = to_tensor2d(std::move(X_test));

    Tensor4D X_train_4d = to_tensor4d(X_train_tensor, 28, 28, 3); // escala de grises convertido a RGB
    Tensor4D X_test_4d = to_tensor4d(X_test_tensor, 28, 28, 3);

    net.set_mlp(nn);

    std::cout << "Training neural network for " << epochs << " epoch(s)...\n";
    if (epochs_train)
        net.train_test(X_train_4d, y_train, X_test_4d, y_test, true, dataset_name, epochs);
    else
        net.train_test(X_train_4d, y_train, X_test_4d, y_test, true, dataset_name);
    // nn.save_data(save_path);
    return 0;
}
