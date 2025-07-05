#include "mlp.hpp"

Mlp::Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
         double lr, std::vector<ActivationType> activation_types, optimizer_type opt,
         bool regularizer, bool dropout)
    : n_inputs(n_inputs), n_outputs(n_outputs), learning_rate(lr)
{
    if (layer_sizes.size() != activation_types.size())
        throw std::invalid_argument("layer_sizes and activation_types must have the same length.");
    int prev_size = n_inputs;
    for (size_t i = 0; i < layer_sizes.size(); ++i)
    {
        std::cout << "Creating layer " << i + 1
                  << " with " << layer_sizes[i]
                  << " neurons and activation " << activation_types[i] << ".\n";
        layers.emplace_back(prev_size, layer_sizes[i], activation_types[i]);
        prev_size = layer_sizes[i];
    }
    if (opt == optimizer_type::RMSPROP)
        this->optimizer = std::make_shared<RMSProp>();
    else if (opt == optimizer_type::ADAM)
        this->optimizer = std::make_shared<Adam>();
    else
        this->optimizer = std::make_shared<SGD>();
    if (regularizer)
    {
        double l2_penalty = 0.01; // Default value, can be adjusted
        this->regularizer = std::make_shared<L2Regularizer>(l2_penalty);
        this->optimizer->set_regularizer(this->regularizer);
        std::cout << "Using L2 regularization: " << l2_penalty << ".\n";
    }
    if (dropout)
    {
        double dropout_rate = 0.2; // Default value, can be adjusted
        this->dropout = std::make_shared<DropoutController>(0.5);
        std::cout << "Using dropout: " << dropout_rate << ".\n";
    }
}

void Mlp::forward_batch(const Tensor2D &input_batch,
                        bool train)
{
    int batch = input_batch.batch_size();
    reusable_activations.clear();
    reusable_activations.emplace_back(input_batch);
    for (size_t l = 0; l < layers.size(); ++l)
    {
        const Tensor2D &prev = reusable_activations.back();
        int out_size = layers[l].get_output_size();
        Tensor2D current(batch, out_size);
        for (int b = 0; b < batch; ++b)
        {
            std::vector<double> output(out_size);
            layers[l].linear_forward(prev.row(b), output);
            layers[l].apply_activation(output);
            if (dropout && train && l < layers.size() - 1)
                dropout->apply(output);
            current.set_row(b, output);
        }
        reusable_activations.emplace_back(std::move(current));
    }
}

void Mlp::backward_batch(const Tensor2D &input_batch,
                         const std::vector<int> &labels)
{
    int batch = input_batch.batch_size();
    const auto &activations = reusable_activations;

    if ((int)reusable_deltas.size() != (int)layers.size())
    {
        reusable_deltas.resize(layers.size());
        for (size_t l = 0; l < layers.size(); ++l)
            reusable_deltas[l] = Tensor2D(batch, layers[l].get_output_size());
    }

    std::vector<double> expected(n_outputs);
    for (int l = (int)layers.size() - 1; l >= 0; --l)
    {
        int out_size = layers[l].get_output_size();
        for (int b = 0; b < batch; ++b)
        {
            double *delta = reusable_deltas[l].row_ptr(b);
            std::fill(delta, delta + out_size, 0.0);
            const std::vector<double> &output = activations[l + 1].row(b);
            one_hot_encode_ptr(labels[b], expected.data(), n_outputs);
            if (layers[l].get_activation() == SOFTMAX)
                for (int i = 0; i < out_size; ++i)
                    delta[i] = output[i] - expected[i];
            else
                for (int i = 0; i < out_size; ++i)
                {
                    double error = 0.0;
                    if (l + 1 < (int)layers.size())
                        for (int j = 0; j < layers[l + 1].get_output_size(); ++j)
                            error += reusable_deltas[l + 1](b, j) * layers[l + 1].get_weight(j, i);
                    if (layers[l].get_activation() == RELU)
                        delta[i] = error * relu_derivative(output[i]);
                    else if (layers[l].get_activation() == SIGMOID)
                        delta[i] = error * sigmoid_derivative(output[i]);
                    else
                        delta[i] = error * tanh_derivative(output[i]);
                }
        }
    }
    for (size_t l = 0; l < layers.size(); ++l)
        for (int b = 0; b < batch; ++b)
            layers[l].apply_update(optimizer, reusable_deltas[l].row(b),
                                   activations[l].row(b), learning_rate, l);
}

void Mlp::train_batch(const Tensor2D &input_batch,
                      const std::vector<int> &labels, double &avg_loss)
{
    forward_batch(input_batch, true);
    const Tensor2D &output_batch = reusable_activations.back();

    double total_loss = 0.0;
    std::vector<double> expected(n_outputs);

    for (int b = 0; b < input_batch.batch_size(); ++b)
    {
        one_hot_encode_ptr(labels[b], expected.data(), n_outputs);
        total_loss += cross_entropy_loss_ptr(output_batch.row_ptr(b), expected.data(), n_outputs);
    }

    backward_batch(input_batch, labels);
    avg_loss = total_loss / input_batch.batch_size();
}

void Mlp::train_with_batches(const Tensor2D &images,
                             const std::vector<int> &labels,
                             double &avg_loss,
                             int batch_size)
{
    int n = images.batch_size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    double total_loss = 0.0;
    double penalty = 0.0;
    int total_batches = (n + batch_size - 1) / batch_size;
    std::cout << "Batch " << 0 << "/" << total_batches << "\r" << std::flush;
    for (int r = 0; r < total_batches; ++r)
    {
        int start = r * batch_size;
        int end = std::min(start + batch_size, n);
        int current_batch_size = end - start;
        Tensor2D batch(current_batch_size, n_inputs);
        std::vector<int> batch_labels(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i)
        {
            batch.set_row(i, images.row(indices[start + i]));
            batch_labels[i] = labels[indices[start + i]];
        }
        std::cout << "Batch " << (r + 1) << "/" << total_batches << "\r" << std::flush;
        double batch_loss;
        train_batch(batch, batch_labels, batch_loss);
        total_loss += batch_loss * current_batch_size;
    }
    if (regularizer)
        penalty = regularizer->compute_penalty(layers);
    avg_loss = (total_loss + penalty) / n;
}

void Mlp::test(const Tensor2D &images,
               const std::vector<int> &labels,
               double &test_accuracy)
{
    int correct = 0;
    forward_batch(images, false);
    const Tensor2D &output_batch = reusable_activations.back();
    for (int i = 0; i < images.batch_size(); ++i)
    {
        const std::vector<double> &output = output_batch.row(i);
        int pred = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (pred == labels[i])
            ++correct;
    }
    test_accuracy = 100.0 * correct / images.batch_size();
}

void Mlp::evaluate(const Tensor2D &images,
                   const std::vector<int> &labels,
                   double &train_accuracy)
{
    int correct = 0;
    forward_batch(images, false);
    const Tensor2D &output_batch = reusable_activations.back();
    for (int i = 0; i < images.batch_size(); ++i)
    {
        const std::vector<double> &output = output_batch.row(i);
        int pred = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (pred == labels[i])
            ++correct;
    }
    train_accuracy = 100.0 * correct / images.batch_size();
}

void Mlp::train_test(const Tensor2D &train_images, const std::vector<int> &train_labels,
                     const Tensor2D &test_images, const std::vector<int> &test_labels,
                     bool Test, const std::string &dataset_filename, int epochs)
{
    std::string base_name = std::filesystem::path(dataset_filename).stem().string();
    std::filesystem::path output_dir = std::filesystem::path("output") / base_name;
    std::filesystem::create_directories(output_dir);
    std::ofstream log_file(output_dir / "log.txt");
    if (!log_file)
    {
        std::cerr << "Can't open '" << output_dir / "log.txt" << "' for writing.\n";
        return;
    }

    int epoch = 0;
    double average_loss = 0, train_accuracy = 0, test_accuracy = 0;
    double best_test_accuracy = -1.0;

    while (true)
    {
        int batch_size = 128;
        auto train_start = std::chrono::high_resolution_clock::now();
        train_with_batches(train_images, train_labels, average_loss, batch_size);
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
        std::ostringstream log_line;
        log_line << "Epoch " << (epoch + 1)
                 << ", Train Loss: " << average_loss
                 << ", Train Acc: " << train_accuracy << "%"
                 << ", Train Time: " << train_time << "s";
        if (Test)
        {
            log_line << ", Test Acc: " << test_accuracy << "%"
                     << ", Test Time: " << test_time << "s";
            if (test_accuracy > best_test_accuracy)
            {
                best_test_accuracy = test_accuracy;
                std::string best_model_path = (output_dir / "best_model.dat").string();
                save_data(best_model_path);
            }
        }
        std::cout << log_line.str() << std::endl;
        log_file << log_line.str() << std::endl;
        if ((epoch + 1) % 10 == 0)
        {
            std::string filename = (output_dir / ("epoch_" + std::to_string(epoch + 1) + ".dat")).string();
            save_data(filename);
            std::cout << "Model saved at epoch " << (epoch + 1) << " to " << filename << ".\n";
        }
        if (average_loss < 1e-5 || epoch >= epochs)
        {
            std::cout << "Stopping training: early stopping criteria met.\n";
            break;
        }
        epoch++;
    }
}

void Mlp::save_data(const std::string &filename) const
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Error: no se pudo abrir " << filename << " para guardar.\n";
        return;
    }

    // Cabecera
    out << n_inputs << " ";
    for (const auto &layer : layers)
        out << layer.get_output_size() << " ";
    out << "\n"
        << learning_rate << "\n";

    // Tipos de activación por capa
    for (const auto &layer : layers)
        out << to_string(layer.get_activation()) << " ";
    out << "\n";

    // Optimizador
    out << to_string(optimizer->get_type()) << "\n";

    // Capas y neuronas
    for (size_t i = 0; i < layers.size(); ++i)
        layers[i].save(out, i);
    out.close();
}

bool Mlp::load_data(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in)
    {
        std::cerr << "Error: no se pudo abrir " << filename << " para leer.\n";
        return false;
    }

    std::cout << "Cargando MLP desde " << filename << "...\n";
    // Leer arquitectura
    std::vector<int> layer_sizes;
    std::string line;
    std::getline(in, line);
    std::istringstream arch_stream(line);
    arch_stream >> n_inputs;
    int size;
    while (arch_stream >> size)
        layer_sizes.push_back(size);
    n_outputs = layer_sizes.back();
    layer_sizes.pop_back();
    in >> learning_rate;

    // Leer funciones de activación
    std::vector<ActivationType> activations(layer_sizes.size() + 1);
    for (size_t i = 0; i < activations.size(); ++i)
    {
        std::string act;
        in >> act;
        activations[i] = from_string(act);
    }
    // Leer optimizador
    std::string opt_type;
    in >> opt_type;
    optimizer_type opt = from_string_opt(opt_type);
    if (opt == optimizer_type::RMSPROP)
        this->optimizer = std::make_shared<RMSProp>();
    else if (opt == optimizer_type::ADAM)
        this->optimizer = std::make_shared<Adam>();
    else
        this->optimizer = std::make_shared<SGD>();

    // Construir capas y cargar datos
    layers.clear();
    int prev_size = n_inputs;
    for (size_t i = 0; i < activations.size(); ++i)
    {
        int curr_size = (i < layer_sizes.size()) ? layer_sizes[i] : n_outputs;
        Layer layer(prev_size, curr_size, activations[i], true);
        layer.load(in);
        layers.push_back(std::move(layer));
        prev_size = curr_size;
    }

    in.close();
    return true;
}
