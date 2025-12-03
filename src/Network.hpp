#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "Layer.hpp"
#include <vector>

#include "Loss.hpp"

class Network {
public:
	std::vector<Layer> layers;

	// For Adam optimizer
	const float beta1 = 0.9f;	// momentum parameter
	const float beta2 = 0.999f; // momentum parameter
	const float epsilon = 1e-8f; // stabilizer
	int t = 0;	// timestep

	Network() {}

	void addLayer(Layer& layer) {
		layers.push_back(std::move(layer));
	}

	const Matrix *forward(const Matrix &input) {
		const Matrix *output = &input;
		for (auto& layer : layers) {
			output = &layer.forward(*output);
		}
		
		return output;
	}

	/**
	* @brief Does a backpropagation pass through the network.
	* Computes and stores the derivations of individuals layers
	* @param gradientOutput Output of a loss derivative function.
	*/
	void backward(const Matrix& gradientOutput) {
		const Matrix *grad = &gradientOutput;

		for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
			grad = &layers[i].backward(*grad);
		}
	}

	/**
	* @brief Go through the network and use the computed gradients
	* to update the network parameters.
	* @param learningRate The size of the gradient step.
	*/
	void updateSGD(float learningRate) {
		for (auto& layer : layers) {
			// Update weight: W = W - lr * dW;
			layer.a_buffer_n_m.scaleInPlace(layer.last_dW, learningRate);
			layer.weights.subtractInPlace(layer.a_buffer_n_m);

			// Update weight: b = b - lr * dB;
			layer.a_buffer_n_1.scaleInPlace(layer.last_dB, learningRate);
			layer.biases.subtractInPlace(layer.a_buffer_n_1);
		}
	}

	void updateAdam(float learningRate) {
		t += 1;

		float biasCorrection1 = 1.0f - powf(beta1, t);
		float biasCorrection2 = 1.0f - powf(beta2, t);

		for (auto& layer : layers) {
			// Update weights
			// --------------------------
			// mW = beta1 * mW + (1 - beta1) * dW;
			layer.mW.scaleInPlace(layer.mW, beta1);
			layer.a_buffer_n_m.scaleInPlace(layer.last_dW, 1.0f - beta1);
			layer.mW.addInPlace(layer.a_buffer_n_m);

			// vW = beta2 * vW + (1 - beta2) * (dW ⊙ dW)
			layer.vW.scaleInPlace(layer.vW, beta2);
			layer.a_buffer_n_m.hadamardInPlace(layer.last_dW, layer.last_dW);
			layer.a_buffer_n_m.scaleInPlace(layer.a_buffer_n_m, 1.0f - beta2);
			layer.vW.addInPlace(layer.a_buffer_n_m);

			// mW_hat = mW / (1 - beta1^t)
			layer.a_buffer_n_m.scaleInPlace(layer.mW, 1.0f / biasCorrection1);

			// vW_hat = vW / (1 - beta2^t)
			layer.b_buffer_n_m.scaleInPlace(layer.vW, 1.0f / biasCorrection2);

			// denom = sqrt(vW_hat) + eps
			layer.b_buffer_n_m.applyFunction(sqrtf);
			layer.b_buffer_n_m.addInPlace(epsilon);

			// update = lr * mW_hat / denom
			layer.a_buffer_n_m.hadamardDivisionInPlace(layer.a_buffer_n_m, layer.b_buffer_n_m);
			layer.a_buffer_n_m.scaleInPlace(layer.a_buffer_n_m, learningRate);
			
			// W = W - update
			layer.weights.subtractInPlace(layer.a_buffer_n_m);

			// Update biases
			// --------------------------
			// mB = beta1 * mB + (1 - beta1) * dB;
			layer.mB.scaleInPlace(layer.mB, beta1);
			layer.a_buffer_n_1.scaleInPlace(layer.last_dB, 1.0f - beta1);
			layer.mB.addInPlace(layer.a_buffer_n_1);

			// vB = beta2 * vB + (1 - beta2) * (dB ⊙ dB)
			layer.vB.scaleInPlace(layer.vB, beta2);
			layer.a_buffer_n_1.hadamardInPlace(layer.last_dB, layer.last_dB);
			layer.a_buffer_n_1.scaleInPlace(layer.a_buffer_n_1, 1.0f - beta2);
			layer.vB.addInPlace(layer.a_buffer_n_1);

			// mB_hat = mB / (1 - beta1^t)
			layer.a_buffer_n_1.scaleInPlace(layer.mB, 1.0f / biasCorrection1);

			// vB_hat = vB / (1 - beta2^t)
			layer.b_buffer_n_1.scaleInPlace(layer.vB, 1.0f / biasCorrection2);

			// denom = sqrt(vB_hat) + eps
			layer.b_buffer_n_1.applyFunction(sqrtf);
			layer.b_buffer_n_1.addInPlace(epsilon);

			// update = lr * mB_hat / denom
			layer.a_buffer_n_1.hadamardDivisionInPlace(layer.a_buffer_n_1, layer.b_buffer_n_1);
			layer.a_buffer_n_1.scaleInPlace(layer.a_buffer_n_1, learningRate);
			
			// b = b - update
			layer.biases.subtractInPlace(layer.a_buffer_n_1);
		}
	}
};


#endif // !NETWORK_HPP
