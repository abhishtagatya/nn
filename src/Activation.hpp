#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>
#include "Matrix.hpp"

inline float relu(float x) { return x > 0 ? x : 0; }
inline float relu_derivative(float x) { return x > 0 ? 1 : 0; }

inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float sigmoid_derivative(float x) { return x * (1 - x); }

/// Allocation free
inline void softmax(Matrix& result, const Matrix& input)
{
    int rows = input.getRows();
    int cols = input.getCols();

    for (int c = 0; c < cols; c++) {

        float maxVal = input(0, c);
        for (int r = 1; r < rows; r++)
            if (input(r, c) > maxVal)
                maxVal = input(r, c);

        float sum_exp = 0.0f;
        for (int r = 0; r < rows; r++)
            sum_exp += expf(input(r, c) - maxVal);

        for (int r = 0; r < rows; r++)
            result(r, c) = expf(input(r, c) - maxVal) / sum_exp;
    }
}


inline Matrix softmax(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    softmax(result, input);
    
    return result;
}

#endif // !ACTIVATION_HPP
