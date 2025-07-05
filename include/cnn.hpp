#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "tensor2d.hpp"
#include "layer_cnn.hpp"

class NeuralNetwork
{
private:
    std::vector<std::unique_ptr<Layer>> layers;
    bool input_is_2d = false;

public:
    void add_layer(std::unique_ptr<Layer> layer)
    {
        layers.push_back(std::move(layer));
    }

    void forward(const Tensor4D &input)
    {
        Tensor4D current4d = input;
        Tensor2D current2d;

        for (auto &layer : layers)
        {
            if (!input_is_2d)
            {
                layer->forward(current4d);
                if (layer->is_2d_output())
                {
                    current2d = layer->output_2d();
                    input_is_2d = true;
                }
                else
                {
                    current4d = layer->output_4d();
                }
            }
            else
            {
                layer->forward(current2d);
                if (layer->is_2d_output())
                    current2d = layer->output_2d();
                else
                    throw std::runtime_error("Can't go from 2D back to 4D");
            }
        }
    }

    const Tensor2D &get_output() const
    {
        if (!layers.empty() && layers.back()->is_2d_output())
            return layers.back()->output_2d();
        else
            throw std::runtime_error("Output is not 2D");
    }
};