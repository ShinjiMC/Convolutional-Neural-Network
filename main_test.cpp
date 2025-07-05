#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "mlp.hpp"
#include "dataset.hpp"

int main(int argc, char *argv[])
{
    std::string model_path = "";
    std::string test_file = "";

    // Leer argumentos
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--save_data" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--test" && i + 1 < argc)
            test_file = argv[++i];
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    if (model_path.empty())
    {
        std::cerr << "Error: --save_data <path_to_model> is required.\n";
        return 1;
    }

    if (test_file.empty())
    {
        std::cerr << "Error: --test <test_dataset.txt> is required.\n";
        return 1;
    }

    // Cargar modelo
    Mlp nn;
    if (!nn.load_data(model_path))
    {
        std::cerr << "Failed to load model from: " << model_path << "\n";
        return 1;
    }
    std::cout << "Model loaded from " << model_path << "\n";

    // Cargar dataset de prueba
    Dataset test(test_file);
    std::vector<std::vector<double>> X_test = test.get_X();
    std::vector<int> y_test = test.get_ys();

    test.print_data("TEST");

    // Ejecutar evaluación
    // nn.test_info(X_test, y_test);

    return 0;
}
