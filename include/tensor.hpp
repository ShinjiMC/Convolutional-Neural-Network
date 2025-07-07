#pragma once
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cassert>

class Tensor4D
{
private:
    int n, c, h, w;
    std::vector<double> data;

public:
    Tensor4D() = default;
    Tensor4D(int n_, int c_, int h_, int w_, double fill = 0.0)
        : n(n_), c(c_), h(h_), w(w_), data(n_ * c_ * h_ * w_, fill) {}

    inline double &operator()(int ni, int ci, int hi, int wi)
    {
        return data[((ni * c + ci) * h + hi) * w + wi];
    }

    inline double operator()(int ni, int ci, int hi, int wi) const
    {
        return data[((ni * c + ci) * h + hi) * w + wi];
    }

    std::tuple<int, int, int, int> shape() const
    {
        return {n, c, h, w};
    }

    int batch_size() const { return n; }
    int channels() const { return c; }
    int height() const { return h; }
    int width() const { return w; }

    void fill(double val)
    {
        std::fill(data.begin(), data.end(), val);
    }

    int size() const { return data.size(); }

    std::vector<double> &flat() { return data; }
    const std::vector<double> &flat() const { return data; }
    Tensor4D batch(int index) const
    {
        if (index < 0 || index >= n)
            throw std::out_of_range("Batch index out of range");

        Tensor4D single(1, c, h, w);
        for (int ci = 0; ci < c; ++ci)
            for (int hi = 0; hi < h; ++hi)
                for (int wi = 0; wi < w; ++wi)
                    single(0, ci, hi, wi) = (*this)(index, ci, hi, wi);
        return single;
    }
};
