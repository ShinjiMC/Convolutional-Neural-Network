#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include "mlp.hpp"
#include "dataset.hpp"
#include "config.hpp"

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

    int n_inputs = X_train[0].size();
    Config cfg;
    if (!cfg.load_config(config_path, n_inputs))
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
        nn = Mlp(n_inputs, cfg.get_layer_sizes(),
                 cfg.get_layer_sizes().back(), cfg.get_learning_rate(),
                 cfg.get_activations(), cfg.get_optimizer(), true, true);
    }

    std::string dataset_name = get_last_folder(dataset_dir);
    save_path = "./output/" + dataset_name + "/final.dat";

    std::cout << "Training neural network for " << epochs << " epoch(s)...\n";
    if (epochs_train)
        nn.train_test(X_train, y_train, X_test, y_test, true, dataset_name, epochs);
    else
        nn.train_test(X_train, y_train, X_test, y_test, true, dataset_name);
    nn.save_data(save_path);
    return 0;
}
