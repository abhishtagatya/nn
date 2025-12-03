#ifndef LOSS_HPP
#define LOSS_HPP
#include <cfloat>

#include "Activation.hpp"
#include "Matrix.hpp"

typedef std::function<float(const Matrix&, const Matrix&)> lossFunction;
typedef std::function<Matrix(const Matrix&, const Matrix&)> lossFunction_derivative;

// --------------------------------------------------
// -- Mean Squared Error (MSE)
// --------------------------------------------------
inline float meanSquaredError(const Matrix& networkOutput, const Matrix& target) {
    float sum = 0.0f;
    const int n = static_cast<int>(networkOutput.getRows());

    for (int i = 0; i < n; ++i) {
        // Using the 1/2n multiplication to make the derivative simpler
        sum += 0.5f * std::pow(networkOutput(i, 0) - target(i, 0), 2);
    }

    return sum;
}

// Allocation-free
inline void meanSquaredError_derivative(Matrix& result, const Matrix& networkOutput, const Matrix& target) {
    assert(result.getRows() == networkOutput.getRows());
    assert(result.getCols() == networkOutput.getCols());
    assert(result.getRows() == target.getRows());
    assert(result.getCols() == target.getCols());

    const int n = static_cast<int>(networkOutput.getRows());
    
    for (int i = 0; i < n; ++i)
        result(i, 0) = (networkOutput(i, 0) - target(i, 0));
}

inline Matrix meanSquaredError_derivative(const Matrix& networkOutput, const Matrix& target) {
    Matrix result(networkOutput.getRows(), 1);
    meanSquaredError_derivative(result, networkOutput, target);
    return result;
}

// --------------------------------------------------
// -- Cross Entropy
// --------------------------------------------------
inline float crossEntropyLoss(const Matrix& networkOutput, const Matrix& target) {
    float loss = 0.0f;

    for (int i = 0; i < networkOutput.getRows(); ++i)
        loss -= target(i, 0) * logf(networkOutput(i, 0) + FLT_EPSILON); // epsilon is supposed to improve stability

    return loss;
}

// Allocation-free
inline void crossEntropyLoss_derivative(Matrix& result, const Matrix& networkOutput, const Matrix& target) {
    assert(result.getRows() == networkOutput.getRows());
    assert(result.getCols() == networkOutput.getCols());
    assert(result.getRows() == target.getRows());
    assert(result.getCols() == target.getCols());
    
    for (int i = 0; i < networkOutput.getRows(); ++i)
        result(i, 0) = networkOutput(i, 0) - target(i, 0);
}

inline Matrix crossEntropyLoss_derivative(const Matrix& networkOutput, const Matrix& target) {
    Matrix result(networkOutput.getRows(), 1);
    crossEntropyLoss_derivative(result, networkOutput, target);
    return result;
}

// --------------------------------------------------
// -- Softmax Cross Entropy
// --------------------------------------------------
inline float softMaxCrossEntropyLoss(const Matrix& networkOutput, const Matrix& target) {
    assert(networkOutput.getRows() == target.getRows());
    assert(networkOutput.getCols() == target.getCols());
    Matrix y_pred = softmax(networkOutput);
    return crossEntropyLoss(y_pred, target);
}

// TODO: add non-alloc version for the softmax
/*

inline void softMaxCrossEntropyLoss_derivative(Matrix& result, const Matrix& networkOutput, const Matrix& target) {
    Matrix y_pred = softmax(networkOutput);
    crossEntropyLoss_derivative(result, networkOutput, target);
}
*/

inline void softMaxCrossEntropyLoss_derivative(Matrix& result, const Matrix& logits, const Matrix& target)
{
    softmax(result, logits);
    // subtract target from each column
    for (int i = 0; i < result.getRows(); i++)
        for (int j = 0; j < result.getCols(); j++)
            result(i, j) -= target(i, j);
}

inline Matrix softMaxCrossEntropyLoss_derivative(const Matrix& logits, const Matrix& target) {
    Matrix result(logits.getRows(), logits.getCols());
    softMaxCrossEntropyLoss_derivative(result, logits, target);
    return result;
}

#endif //LOSS_HPP
