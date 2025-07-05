#pragma once
#include <vector>
#include <stdexcept>

class Tensor2D
{
private:
    int batch_size_, feature_size_;
    std::vector<double> data;

public:
    Tensor2D() : batch_size_(0), feature_size_(0) {}

    Tensor2D(int batch_size, int feature_size)
        : batch_size_(batch_size), feature_size_(feature_size),
          data(batch_size * feature_size, 0.0) {}

    double &operator()(int batch, int feature)
    {
        return data[batch * feature_size_ + feature];
    }

    double operator()(int batch, int feature) const
    {
        return data[batch * feature_size_ + feature];
    }

    void resize(int batch_size, int feature_size)
    {
        batch_size_ = batch_size;
        feature_size_ = feature_size;
        data.resize(batch_size * feature_size, 0.0);
    }

    int batch_size() const { return batch_size_; }
    int feature_size() const { return feature_size_; }

    std::vector<double> &raw_data() { return data; }
    const std::vector<double> &raw_data() const { return data; }

    std::vector<double> row(int batch) const
    {
        std::vector<double> out(feature_size_);
        for (int i = 0; i < feature_size_; ++i)
            out[i] = (*this)(batch, i);
        return out;
    }

    double *row_ptr(int batch)
    {
        return &data[batch * feature_size_];
    }

    const double *row_ptr(int batch) const
    {
        return &data[batch * feature_size_];
    }

    void set_row(int batch, const std::vector<double> &values)
    {
        if ((int)values.size() != feature_size_)
            throw std::runtime_error("Invalid row size in set_row()");
        for (int i = 0; i < feature_size_; ++i)
            (*this)(batch, i) = values[i];
    }
};
