#include "dataset.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <algorithm>

int reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_images(const std::string &filename, std::vector<std::vector<double>> &images, int num_images)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file " + filename);
    int magic, n, rows, cols;
    file.read((char *)&magic, 4);
    magic = reverse_int(magic);
    file.read((char *)&n, 4);
    n = reverse_int(n);
    file.read((char *)&rows, 4);
    rows = reverse_int(rows);
    file.read((char *)&cols, 4);
    cols = reverse_int(cols);
    images.resize(num_images, std::vector<double>(rows * cols));
    for (int i = 0; i < num_images; ++i)
        for (int j = 0; j < rows * cols; ++j)
        {
            unsigned char pixel;
            file.read((char *)&pixel, 1);
            images[i][j] = pixel / 255.0;
        }
}

void read_mnist_labels(const std::string &filename, std::vector<int> &labels, int num_labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file " + filename);
    int magic, n;
    file.read((char *)&magic, 4);
    magic = reverse_int(magic);
    file.read((char *)&n, 4);
    n = reverse_int(n);
    labels.resize(num_labels);
    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char label;
        file.read((char *)&label, 1);
        labels[i] = label;
    }
}

Dataset::Dataset(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cant open file: " + filename);

    int n_inputs, n_outputs;
    file >> n_inputs >> n_outputs;
    if (n_inputs < 1 || n_outputs < 1)
        throw std::runtime_error("Invalid number of inputs or options in file: " + filename);
    // std::cout << "n_inputs: " << n_inputs << ", n_outputs: " << n_outputs << std::endl;
    X.clear();
    y.clear();

    std::vector<double> input_row(n_inputs);
    std::vector<double> output_row(n_outputs);

    while (file)
    {
        for (int i = 0; i < n_inputs; ++i)
            if (!(file >> input_row[i]))
                return;

        for (int j = 0; j < n_outputs; ++j)
            if (!(file >> output_row[j]))
                throw std::runtime_error("Unexpected end of file while reading output vector");

        X.push_back(input_row);
        y.push_back(output_row);
    }
    file.close();
}

const std::vector<std::vector<double>> &Dataset::get_X() const
{
    return X;
}

const std::vector<std::vector<double>> &Dataset::get_y() const
{
    return y;
}

const std::vector<int> &Dataset::get_ys() const
{
    static std::vector<int> ys;
    ys.clear();
    for (const auto &output_row : y)
    {
        int max_index = std::distance(output_row.begin(), std::max_element(output_row.begin(), output_row.end()));
        ys.push_back(max_index);
    }
    return ys;
}

void Dataset::print_data(std::string name) const
{
    std::cout << "==========================\n";
    std::cout << "DATASET " << name << "\n";
    std::cout << "\tTotal samples: " << X.size() << "\n";
    std::cout << "\tInputs per sample: " << (X.empty() ? 0 : X[0].size()) << "\n";
    std::cout << "\tOutputs per sample: " << (y.empty() ? 0 : y[0].size()) << "\n";
    std::cout << "==========================\n\n";
}

void Dataset::generate_mnist(const std::string &filename, int TRAIN_SAMPLES, int TEST_SAMPLES)
{
    std::vector<std::vector<double>> train_images, test_images;
    std::vector<int> train_labels, test_labels;
    read_mnist_images("./archive/train-images.idx3-ubyte", train_images, TRAIN_SAMPLES);
    read_mnist_labels("./archive/train-labels.idx1-ubyte", train_labels, TRAIN_SAMPLES);
    read_mnist_images("./archive/t10k-images.idx3-ubyte", test_images, TEST_SAMPLES);
    read_mnist_labels("./archive/t10k-labels.idx1-ubyte", test_labels, TEST_SAMPLES);

    // Crear archivos train.txt y test.txt
    std::ofstream train_file(filename + "train.txt");
    std::ofstream test_file(filename + "test.txt");

    if (!train_file || !test_file)
        throw std::runtime_error("No se pudieron crear archivos de salida train/test");

    int n_inputs = train_images[0].size();
    int n_outputs = 10;

    // Escribir encabezados
    train_file << n_inputs << " " << n_outputs << "\n";
    test_file << n_inputs << " " << n_outputs << "\n";

    // Guardar datos de entrenamiento
    for (size_t i = 0; i < train_images.size(); ++i)
    {
        for (double v : train_images[i])
            train_file << v << " ";
        for (int j = 0; j < n_outputs; ++j)
            train_file << (train_labels[i] == j ? 1 : 0) << " ";
        train_file << "\n";
    }

    // Guardar datos de test
    for (size_t i = 0; i < test_images.size(); ++i)
    {
        for (double v : test_images[i])
            test_file << v << " ";
        for (int j = 0; j < n_outputs; ++j)
            test_file << (test_labels[i] == j ? 1 : 0) << " ";
        test_file << "\n";
    }

    train_file.close();
    test_file.close();
}