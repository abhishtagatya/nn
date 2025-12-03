#ifndef LAYER_HPP
#define LAYER_HPP
#include "Matrix.hpp"
#include <functional>
#include <utility>

enum LayerRandomizeMethod {
	UNIFORM,
	HE,
	XAVIER,
	NONE
};

class Layer {
public:
	Matrix weights;
	Matrix biases;
	std::function<float(float)> activation;
	std::function<float(float)> activation_d;

	Matrix last_input;
	Matrix last_Z; // output of the layer
	Matrix last_output; // activation function applied on the layer output

	Matrix last_dW;	// derived output wrt weights
	Matrix last_dB; // derived output wrt bias
	Matrix last_dZ; // dL/dZ
	Matrix last_dInput; // derived output wrt input

	// For Adam optimizer
	Matrix mW; // 1st moment for weights
	Matrix vW; // 2nd moment for weights
	Matrix mB; // 1st moment for biases
	Matrix vB; // 2nd moment for biases
	
	Matrix a_buffer_n_m;	// a free-to-use pre-allocated matrix of size n×m
	Matrix b_buffer_n_m;	// a free-to-use pre-allocated matrix of size n×m
	Matrix a_buffer_n_1;	// a free-to-use pre-allocated matrix of size n×1
	Matrix b_buffer_n_1;	// a free-to-use pre-allocated matrix of size n×1

	Layer(size_t input_size, size_t output_size,
		  std::function<float(float)> activation_func,
		  std::function<float(float)> activation_deriv,
		  float weight_init_min = -0.1f,
		  float weight_init_max = 0.1f,
		LayerRandomizeMethod randomize_method = UNIFORM
		)
		: weights(output_size, input_size),
		  biases(output_size, 1),
		  activation(std::move(activation_func)),
		  activation_d(std::move(activation_deriv)),
		  last_input(input_size, 1),
		  last_Z(output_size, 1),
		  last_output(output_size, 1),
		  last_dW(output_size, input_size),
		  last_dB(output_size, 1),
		  last_dZ(output_size, 1),
		  last_dInput(input_size, 1),
		  mW(output_size, input_size),
		  vW(output_size, input_size),
		  mB(output_size, 1),
		  vB(output_size, 1),
		  a_buffer_n_m(output_size, input_size),
		  b_buffer_n_m(output_size, input_size),
		  a_buffer_n_1(output_size, 1),
		  b_buffer_n_1(output_size, 1) {
		switch (randomize_method) {
		case UNIFORM:
			weights.randomize(weight_init_min, weight_init_max);
			break;
		case HE:
			weights.randomizeHe(input_size);
			break;
		case XAVIER:
			weights.randomizeXavier(input_size, output_size);
			break;
		default:
			break;
		}
		biases.fill(0.0f);
	}

	/*
	Matrix& forward(const Matrix& input) {
		Matrix::copy(last_input, input);

		// Z = W * X + b
		Matrix::multiplyInPlace(last_Z, weights, input);
		last_Z.addInPlace(biases);

		// Apply the activation function
		Matrix::applyFunction(last_output, last_Z, activation);
		
		return last_output;
	}*/

	Matrix& forward(const Matrix& input) {
		// Resize buffers for batch processing
		last_input = input; // input_size x batch_size
		last_Z = Matrix(weights.getRows(), input.getCols()); // output_size x batch_size
		last_output = Matrix(weights.getRows(), input.getCols()); // output_size x batch_size

		// Z = W * X + b
		Matrix::multiplyInPlace(last_Z, weights, input); // output_size x batch_size
		last_Z.addInPlace(biases); // broadcasting biases addition

		// Apply the activation function
		Matrix::applyFunction(last_output, last_Z, activation);

		return last_output;
	}

	/*
	Matrix backward(const Matrix& derived_output) {
		// Apply derived activation function
		Matrix::applyFunction(last_dZ, last_output, activation_d);
		Matrix::hadamardInPlace(last_dZ, last_dZ, derived_output);

		// Compute gradients wrt weights and biases
		Matrix::multiplyInPlace(last_dW, last_dZ, Matrix::transpose(last_input));
		Matrix::copy(last_dB, last_dZ); // must be sum across the whole batch

		// Compute gradient wrt input
		Matrix::multiplyInPlace(last_dInput, Matrix::transpose(weights), last_dZ);
		
		return last_dInput;
	}
	*/

	Matrix& backward(const Matrix& derived_output) {
		// Resize buffers for batch processing
		last_dZ = Matrix(last_output.getRows(), derived_output.getCols()); // output_size x batch_size
		last_dInput = Matrix(weights.getCols(), derived_output.getCols()); // input_size x batch_size

		// Apply derived activation function
		Matrix::applyFunction(last_dZ, last_output, activation_d);
		Matrix::hadamardInPlace(last_dZ, last_dZ, derived_output);

		// Compute gradients wrt weights and biases
		Matrix::multiplyInPlace(last_dW, last_dZ, Matrix::transpose(last_input));

		last_dB.fill(0.0f); // Reset last_dB to zero
		for (size_t i = 0; i < last_dZ.getRows(); ++i) {
			for (size_t j = 0; j < last_dZ.getCols(); ++j) {
				last_dB(i, 0) += last_dZ(i, j); // Sum each column of last_dZ into last_dB
			}
		}

		// Compute gradient wrt input
		Matrix::multiplyInPlace(last_dInput, Matrix::transpose(weights), last_dZ);
		
		return last_dInput;
	}
};

#endif // !LAYER_HPP
