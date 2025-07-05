#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <string>
#include <memory>
#include "regularizer.hpp"

enum class optimizer_type
{
    SGD,
    ADAM,
    RMSPROP
};

class Optimizer
{
protected:
    std::shared_ptr<Regularizer> regularizer = nullptr;

public:
    virtual void update(double learning_rate, std::vector<double> &weights, double &bias,
                        const double *input, double delta,
                        int input_size, int neuron_index) = 0;
    virtual optimizer_type get_type() const = 0;
    void set_regularizer(std::shared_ptr<Regularizer> reg)
    {
        this->regularizer = reg;
    }
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer
{
public:
    void update(double learning_rate, std::vector<double> &weights, double &bias,
                const double *input, double delta,
                int input_size, int) override
    {
        double lambda = (this->regularizer ? this->regularizer->lambda_value() : 0.0);
        for (int j = 0; j < input_size; ++j)
        {
            double grad = delta * input[j];
            if (lambda > 0.0)
                grad += lambda * weights[j];
            weights[j] -= learning_rate * grad;
        }
        bias -= learning_rate * delta;
    }
    optimizer_type get_type() const override { return optimizer_type::SGD; }
};

class RMSProp : public Optimizer
{
private:
    double tau;
    double epsilon;
    std::unordered_map<int, std::vector<double>> r_w;
    std::unordered_map<int, double> r_b;

public:
    RMSProp(double tau = 0.99, double epsilon = 1e-8)
        : tau(tau), epsilon(epsilon) {}

    void update(double learning_rate, std::vector<double> &weights, double &bias,
                const double *input, double delta, int input_size, int neuron_index) override
    {
        auto &r_weights = r_w[neuron_index];
        auto &r_bias = r_b[neuron_index];

        if (r_weights.size() != static_cast<size_t>(input_size))
            r_weights.assign(input_size, 0.0);
        double lambda = (this->regularizer ? this->regularizer->lambda_value() : 0.0);
        for (int j = 0; j < input_size; ++j)
        {
            double grad = delta * input[j];
            if (lambda > 0.0)
                grad += lambda * weights[j];
            r_weights[j] = tau * r_weights[j] + (1.0 - tau) * (grad * grad);
            weights[j] -= learning_rate * grad / (std::sqrt(r_weights[j]) + epsilon);
        }

        double grad_b = delta;
        r_bias = tau * r_bias + (1.0 - tau) * (grad_b * grad_b);
        bias -= learning_rate * grad_b / (std::sqrt(r_bias) + epsilon);
    }
    optimizer_type get_type() const override { return optimizer_type::RMSPROP; }
};

class Adam : public Optimizer
{
private:
    double beta1;
    double beta2;
    double epsilon;
    std::unordered_map<int, std::vector<double>> m_w, v_w;
    std::unordered_map<int, double> m_b, v_b;
    std::unordered_map<int, int> timestep;

    std::unordered_map<int, double> beta1_pow_t;
    std::unordered_map<int, double> beta2_pow_t;

public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    void update(double learning_rate, std::vector<double> &weights, double &bias,
                const double *input, double delta, int input_size, int neuron_index) override
    {
        auto &m_weights = m_w[neuron_index];
        auto &v_weights = v_w[neuron_index];
        auto &m_bias = m_b[neuron_index];
        auto &v_bias = v_b[neuron_index];
        auto &t = timestep[neuron_index];
        auto &b1_pow = beta1_pow_t[neuron_index];
        auto &b2_pow = beta2_pow_t[neuron_index];

        if (m_weights.size() != static_cast<size_t>(input_size))
        {
            m_weights.assign(input_size, 0.0);
            v_weights.assign(input_size, 0.0);
            b1_pow = beta1;
            b2_pow = beta2;
            t = 1;
        }
        else
        {
            t += 1;
            b1_pow *= beta1;
            b2_pow *= beta2;
        }

        double correction1 = 1.0 / (1.0 - b1_pow);
        double correction2 = 1.0 / (1.0 - b2_pow);
        double lambda = (this->regularizer ? this->regularizer->lambda_value() : 0.0);

        for (int j = 0; j < input_size; ++j)
        {
            double grad = delta * input[j];
            if (lambda > 0.0)
                grad += lambda * weights[j];
            m_weights[j] = beta1 * m_weights[j] + (1.0 - beta1) * grad;
            v_weights[j] = beta2 * v_weights[j] + (1.0 - beta2) * grad * grad;

            double m_hat = m_weights[j] * correction1;
            double v_hat = v_weights[j] * correction2;

            weights[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        double grad_b = delta;
        m_bias = beta1 * m_bias + (1.0 - beta1) * grad_b;
        v_bias = beta2 * v_bias + (1.0 - beta2) * grad_b * grad_b;

        double m_hat_b = m_bias * correction1;
        double v_hat_b = v_bias * correction2;

        bias -= learning_rate * m_hat_b / (std::sqrt(v_hat_b) + epsilon);
    }
    optimizer_type get_type() const override { return optimizer_type::ADAM; }
};

inline const std::unordered_map<optimizer_type, std::string> opt_to_string = {
    {optimizer_type::SGD, "sgd"},
    {optimizer_type::ADAM, "adam"},
    {optimizer_type::RMSPROP, "rmsprop"}};

inline const std::unordered_map<std::string, optimizer_type> string_to_opt = {
    {"sgd", optimizer_type::SGD},
    {"adam", optimizer_type::ADAM},
    {"rmsprop", optimizer_type::RMSPROP}};

inline std::string to_string(optimizer_type type)
{
    auto it = opt_to_string.find(type);
    return it != opt_to_string.end() ? it->second : "unknown";
}

inline optimizer_type from_string_opt(const std::string &str)
{
    auto it = string_to_opt.find(str);
    return it != string_to_opt.end() ? it->second : optimizer_type::SGD;
}