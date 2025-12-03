#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <random>
#include <vector>
#include <cassert>
#include <functional>
#include <cstring>
#include <iostream>

#include "DataLoader.hpp"

class Matrix {
public:
    std::vector<float> values;

    // --------------------------------------------------
    // -- Constructors
    // --------------------------------------------------
    
    Matrix(const size_t rows, const size_t cols) : values(rows * cols), rows(rows), cols(cols) {}
    
    Matrix(const Matrix& matrix) : values(matrix.values), rows(matrix.rows), cols(matrix.cols) {}

    // --------------------------------------------------
    // -- Getters/Setters
    // --------------------------------------------------

    const size_t getRows() const {
        return rows;
	}

    const size_t getCols() const {
		return cols;
	}

    // --------------------------------------------------
    // -- Operators
    // --------------------------------------------------

    inline float& operator()(const size_t row, const size_t col) {
        return values[row * cols + col];
    }

    inline const float& operator()(const size_t row, const size_t col) const {
        return values[row * cols + col];
    }

    // --------------------------------------------------
    // -- Instance methods
    // --------------------------------------------------
    /**
    * @brief Computes a product of two matrices.
    * Saves the result into this Matrix.
    * @param left Left matrix.
    * @param right Right matrix.
    */
    inline void multiplyInPlace(const Matrix& left, const Matrix& right) {
        multiplyInPlace(*this, left, right);
    }

    /**
    * @brief Computes a member-wise matrix multiplication (Hadamard product).
    * Saves the result into this Matrix.
    * @param left Left matrix.
    * @param right Right scale.
    */
    inline void hadamardInPlace(const Matrix& left, const Matrix& right) {
        hadamardInPlace(*this, left, right);
    }

    /**
    * @brief Computes a member-wise matrix division (Hadamard division).
    * Saves the result into this Matrix.
    * @param left Left matrix.
    * @param right Right scale.
    */
    inline void hadamardDivisionInPlace(const Matrix& left, const Matrix& right) {
        hadamardDivisionInPlace(*this, left, right);
    } 

    /**
    * @brief Computes a matrix multiplication by scalar.
    * Saves the result into this Matrix.
    * @param left Left matrix.
    * @param scalar Scalar scale.
    */
    inline void scaleInPlace(const Matrix& left, const float scalar) {
        scaleInPlace(*this, left, scalar);
    }
    
    /**
    * @brief Fills the matrix with set value.
    * @param value Value to be used to fill the Matrix.
    */
    inline void fill(const float value) {
        for (int i = 0; i < rows * cols; ++i) {
            values[i] = value;
        }
    }

    /**
    * @brief Fills the matrix with random values.
    * @param min Minimum random value.
    * @param max Maximum random value.
    */
    inline void randomize(const float min, const float max) {
        static std::mt19937 generator(42);  // DETERMINISTIC RANDOM
        std::uniform_real_distribution<float> distribution(min, max);
        for (int i = 0; i < rows * cols; ++i) {
            values[i] = distribution(generator);
        }
    }

    /*
	* @brief Fills the matrix with He-initialized random values.
	* @param fan_in Number of input units.
    */
    void randomizeHe(size_t fan_in) {
        float stddev = std::sqrt(2.0f / fan_in);
        static std::mt19937 generator(42);  // DETERMINISTIC RANDOM
        std::normal_distribution<float> distribution(0.0f, stddev);
        for (int i = 0; i < rows * cols; ++i) {
            values[i] = distribution(generator);
		}
    }
    
    /*
	* @brief Fills the matrix with Xavier-initialized random values.
	* @param fan_in Number of input units.
    */
    void randomizeXavier(size_t fan_in, size_t fan_out) {
        float stddev = std::sqrt(1.0f / ((fan_in + fan_out) / 2.0f));
        static std::mt19937 generator(42);  // DETERMINISTIC RANDOM
        std::normal_distribution<float> distribution(0.0f, stddev);
        for (int i = 0; i < rows * cols; ++i) {
            values[i] = distribution(generator);
        }
    }

    /**
    * @brief Applies a (float -> float) function to every member of this Matrix.
    * Saves the result into this Matrix.
    * @param func Function to be applied.
    */
    inline void applyFunction(const std::function<float(float)>& func) {
        for (int i = 0; i < rows * cols; ++i) {
            values[i] = func(values[i]);
        }
	}

    /**
    * @brief Adds this and the given Matrix.
    * Saves the result into this Matrix.
    * @param right Matrix to add.
    */
    inline void addInPlace(const Matrix& right) {
        // Ensure dimensions are compatible
        if (this->getRows() == right.getRows() && right.getCols() == 1) {
            // Broadcasting: Add each element of `right` to every column of `this`
            for (size_t col = 0; col < this->getCols(); ++col) {
                for (size_t row = 0; row < this->getRows(); ++row) {
                    (*this)(row, col) += right(row, 0);
                }
            }
        }
        else if (this->getRows() == right.getRows() && this->getCols() == right.getCols()) {
            // Element-wise addition
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] += right.values[i];
            }
        }
        else {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }
    }

    /**
    * @brief Adds this Matrix and the given scalar.
    * Saves the result into this Matrix.
    * @param scalar Scalar to add.
    */
    inline void addInPlace(const float scalar) {
        for (int i = 0; i < rows * cols; ++i) {
            values[i] += scalar;
        }
    }

    /**
    * @brief Subtracts the given Matrix from this Matrix.
    * Saves the result into this Matrix.
    * @param right Matrix to subtract.
    */
    inline void subtractInPlace(const Matrix& right) {
        for (int i = 0; i < rows * cols; ++i) {
            values[i] -= right.values[i];
        }
    }

    /**
    * @brief Transpose this square Matrix.
    * Saves the result into this Matrix.
    * @note Only works for square Matrices.
    */
    inline void transposeSquareInPlace() {
        assert(rows == cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = i + 1; j < cols; ++j) {
                std::swap((*this)(i, j), (*this)(j, i));
            }
        }
    }

    /**
    * @brief Copies a row from this Matrix and saves into target Matrix.
    * @param rowIndex Index of the row to copy.
    * @param target Matrix into which to copy the row.
    * Must be a single-row Matrix with number of columns same as this Matrix.
    */
    inline void getRow(const int rowIndex, Matrix& target) {
        assert(target.rows == 1);
        assert(target.cols == cols);

        for (int col = 0; col < cols; ++col) {
            target(0, col) = (*this)(rowIndex, col);
        }
    }

    /**
    * @brief Copies a column from this Matrix and saves into target Matrix.
    * @param columnIndex Index of the column to copy.
    * @param target Matrix into which to copy the column.
    * Must be a single-column Matrix with number of rows same as this Matrix.
    */
    inline void getColumn(const int columnIndex, Matrix& target) {
        assert(target.cols == 1);
        assert(target.rows == rows);

        for (int row = 0; row < rows; ++row) {
            target(row, 0) = (*this)(row, columnIndex);
        }
    }
    
    void debugPrint() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << values[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // --------------------------------------------------
    // -- Static methods
    // --------------------------------------------------
    /**
    * @brief Initializes a Matrix from CSV data.
    * @param data CSV data.
    * @returns New Matrix from CSV.
    */
    static inline Matrix fromCSV(const csvFile &data) {
        const size_t rows = data.size();
        const size_t cols = data[0].size();
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j];
            }
        }

        return result;
    }
    
    /**
    * @brief Copies the source Matrix into given target.
    * @param target Target Matrix to copy source into.
    * @param source Matrix to copy.
    */
    static inline void copy(Matrix& target, const Matrix& source) {
        assert(source.rows == target.rows);
        assert(source.cols == target.cols);

        memcpy(target.values.data(), source.values.data(), sizeof(float) * source.rows * source.cols);
    }

    /**
    * @brief Returns a product of two matrices.
    * @param left Left matrix.
    * @param right Right matrix.
    * @returns Left * Right result.
    *
    * @note Allocates a new matrix. Use @link multiplyInPlace instead unless necessary.
    */
    static inline Matrix multiply(const Matrix& left, const Matrix& right) {
        assert(left.cols == right.rows);
        Matrix result(left.rows, right.cols);

        multiplyInPlace(result, left, right);

        return result;
    }

    /**
    * @brief Computes a product of two matrices.
    * Saves the result into the result Matrix.
    * @param result Matrix to store the result into.
    * @param left Left matrix.
    * @param right Right matrix.
    */
    static inline void multiplyInPlace(Matrix& result, const Matrix& left, const Matrix& right) {
        assert(left.cols == right.rows);
        assert(result.rows == left.rows);
        assert(result.cols == right.cols);

        result.fill(0.0f);
        
        for (int i = 0; i < left.rows; ++i) {
            for (int k = 0; k < left.cols; ++k) {
                const float left_scalar = left(i, k);
                const float* left_row = &right.values[k * right.cols];
                float* result_row = &result.values[i * result.cols];
                
                for (int j = 0; j < right.cols; ++j) {
                    result_row[j] += left_scalar * left_row[j];
                }
            }
        }
    }

    /**
    * @brief Computes a matrix multiplication by scalar.
    * Saves the result into the result Matrix.
    * @param result Matrix to store the result into.
    * @param left Left matrix.
    * @param scalar Scalar scale.
    */
    static inline void scaleInPlace(Matrix& result, const Matrix& left, const float scalar) {
        assert(result.rows == left.rows);
        assert(result.cols == left.cols);
        
        for (int i = 0; i < result.rows * result.cols; ++i) {
            result.values[i] = left.values[i] * scalar;
        }
    }

    /**
    * @brief Computes a member-wise matrix multiplication (Hadamard product).
    * Saves the result into the result Matrix.
    * @param result Matrix to store the result into.
    * @param left Left matrix.
    * @param right Right scale.
    */
    static inline void hadamardInPlace(Matrix& result, const Matrix& left, const Matrix& right) {
        assert(result.rows == left.rows);
        assert(result.cols == left.cols);
        assert(result.rows == right.rows);
        assert(result.cols == right.cols);
        
        for (int i = 0; i < result.rows * result.cols; ++i) {
            result.values[i] = left.values[i] * right.values[i];
        }
    }

    /**
    * @brief Computes a member-wise matrix division (Hadamard division).
    * Saves the result into the result Matrix.
    * @param result Matrix to store the result into.
    * @param left Left matrix.
    * @param right Right scale.
    */
    static inline void hadamardDivisionInPlace(Matrix& result, const Matrix& left, const Matrix& right) {
        assert(result.rows == left.rows);
        assert(result.cols == left.cols);
        assert(result.rows == right.rows);
        assert(result.cols == right.cols);
        
        for (int i = 0; i < result.rows * result.cols; ++i) {
            result.values[i] = left.values[i] / right.values[i];
        }
    }

    /**
    * @brief Create a transposed Matrix.
    * @param matrix Matrix to transpose.
    * @returns New Matrix.
    */
    static inline Matrix transpose(const Matrix& matrix) {
        Matrix result(matrix.cols, matrix.rows);
        for (int i = 0; i < matrix.rows; ++i) {
            for (int j = 0; j < matrix.cols; ++j) {
                result(j, i) = matrix(i, j);
            }
        }
        
        return result;
    }

    /**
    * @brief Applies a (float -> float) function to every member of the target Matrix.
    * Saves the result into the result Matrix.
    * @param result Matrix to save the result into.
    * @param target Matrix to apply the function on.
    * @param func Function to be applied.
    */
    static inline void applyFunction(Matrix& result, const Matrix& target, const std::function<float(float)>& func) {
        assert(result.rows == target.rows);
        assert(result.cols == target.cols);

        for (int i = 0; i < result.rows * result.cols; ++i) {
            result.values[i] = func(target.values[i]);
        }
    }
    
private:
    size_t rows;
	size_t cols;
};

#endif //MATRIX_HPP
