#pragma once
#include <vector>
#include <cstdlib>
#include <random>

class DropoutController
{
private:
    double dropout_rate;
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> dist;

public:
    explicit DropoutController(double rate)
        : dropout_rate(rate), rng(std::random_device{}()), dist(0.0, 1.0) {}

    void apply(std::vector<double> &activations) const
    {
        for (double &val : activations)
        {
            if (dist(rng) < dropout_rate)
                val = 0.0;
            else
                val *= (1.0 / (1.0 - dropout_rate));
        }
    }

    void scale_for_inference(std::vector<double> &activations) const
    {
        for (double &val : activations)
            val *= (1.0 - dropout_rate);
    }

    double rate() const { return dropout_rate; }
};
