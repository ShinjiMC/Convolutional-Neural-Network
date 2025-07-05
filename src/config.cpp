#include "config.hpp"
bool Config::load_config(const std::string &filename, const int inputs)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // Layer Inputs
    if (inputs <= 0)
    {
        std::cerr << "Error: Invalid number of inputs: " << inputs << std::endl;
        return false;
    }
    this->layer_sizes.clear();
    this->activations.clear();
    this->n_inputs = inputs;

    std::string line;
    std::istringstream iss;

    // Leer layer sizes (ocultas + salida)
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read layer sizes line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    int layer_size;
    while (iss >> layer_size)
        this->layer_sizes.push_back(layer_size);

    if (this->layer_sizes.size() < 2)
    {
        std::cerr << "Error: Must specify at least one hidden layer and output layer.\n";
        return false;
    }

    // Leer learning rate
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read learning rate line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    if (!(iss >> this->learning_rate))
    {
        std::cerr << "Error: Invalid learning rate.\n";
        return false;
    }
    int expected_count = static_cast<int>(this->layer_sizes.size());

    // Leer funciones de activación
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read activation functions line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    for (int i = 0; i < expected_count; ++i)
    {
        std::string act_name;
        if (!(iss >> act_name))
        {
            std::cerr << "Error: Not enough activation functions.\n";
            return false;
        }
        this->activations.push_back(from_string(act_name));
    }

    if (activations.size() != expected_count)
    {
        std::cerr << "Error: Mismatch in number of activations.\n";
        return false;
    }

    // Leer optimizador
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read optimizer line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    std::string opt_name;
    if (!(iss >> opt_name))
    {
        std::cerr << "Error: Invalid optimizer name.\n";
        return false;
    }
    this->opt = from_string_opt(opt_name);
    return true;
}

const void Config::print_config()
{
    std::cout << "==========================\n";
    std::cout << "CONFIGURATION\n";
    std::cout << "Layer Sizes: \n";
    std::cout << "\t" << n_inputs << " inputs\n";
    for (const auto &size : layer_sizes)
        std::cout << "\t" << size << " neurons\n";
    std::cout << "Learning Rate: " << learning_rate << "\n";
    std::cout << "Activations: \n";
    for (const auto &act : activations)
        std::cout << "\t" << to_string(act) << "\n";
    std::cout << "Optimizer: " << to_string(opt) << "\n";
    std::cout << "==========================\n";
    std::cout << std::endl;
}