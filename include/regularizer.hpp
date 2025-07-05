#pragma once
#include <vector>
class Layer;

class Regularizer
{
public:
    virtual double compute_penalty(const std::vector<Layer> &layers) const = 0;
    virtual void apply(std::vector<double> &weights, std::vector<double> &grad_weights) const = 0;
    virtual ~Regularizer() = default;
    virtual double lambda_value() const { return 0.0; }
};

class L2Regularizer : public Regularizer
{
private:
    double lambda;

public:
    explicit L2Regularizer(double lambda) : lambda(lambda) {}
    double compute_penalty(const std::vector<Layer> &layers) const override;
    void apply(std::vector<double> &weights, std::vector<double> &grad_weights) const override;
    double lambda_value() const { return lambda; }
};
