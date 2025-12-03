#include <iostream>
#include <filesystem>
#include <algorithm>

#include "DataLoader.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"
#include "Activation.hpp"
#include "Network.hpp"
#include "Time.hpp"

void TestFMNIST(int epochs, float learningRate, int batchSize) {
	constexpr float WEIGHT_INIT_MIN = -0.1f;
	constexpr float WEIGHT_INIT_MAX = 0.1f;

	std::cout << get_time() << ": Loading FMNIST dataset..." << std::endl;
	const csvFile trainData = loadCSV("data/fashion_mnist_train_vectors.csv", 255);
	const csvFile trainLabels = loadCSV("data/fashion_mnist_train_labels.csv");
	const csvFile testData = loadCSV("data/fashion_mnist_test_vectors.csv", 255);
	const csvFile testLabels = loadCSV("data/fashion_mnist_test_labels.csv");

	// Transform data
	Matrix input = Matrix::fromCSV(trainData); // 60000×784
	Matrix input_T = Matrix::transpose(input); // 784×60000
	Matrix output = Matrix::fromCSV(trainLabels);	// 60000×1
	Matrix output_T = Matrix::transpose(output); // 1×60000

	Matrix testInput = Matrix::fromCSV(testData);
	Matrix testInput_T = Matrix::transpose(testInput);
	Matrix testOutput = Matrix::fromCSV(testLabels);
	Matrix testOutput_T = Matrix::transpose(testOutput);

	std::string trainDumpFile = "train_predictions.csv";
	std::string testDumpFile = "test_predictions.csv";

	// Set up the network
	Network network;
	Layer h1(784, 256, relu, relu_derivative, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX, HE);
	Layer h2(256, 128, relu, relu_derivative, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX, HE);
	Layer outputLayer(128, 10, [](float x) { return x; }, [](float x) { return 1.0; }, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX, XAVIER);

	network.addLayer(h1);
	network.addLayer(h2);
	network.addLayer(outputLayer);

	// Pre-allocate stuff
	Matrix inputBatchStandard(784, batchSize);
	Matrix outputBatchStandard(10, batchSize);

	int lastBatchSize = static_cast<int>(input_T.getCols()) % batchSize;
	Matrix inputBatchRest(784, lastBatchSize);
	Matrix outputBatchRest(10, lastBatchSize);

	int lastBatchSizeTest = static_cast<int>(testInput_T.getCols()) % batchSize;
	Matrix inputBatchRestTest(784, lastBatchSizeTest);
	Matrix outputBatchRestTest(10, lastBatchSizeTest);

	Matrix batchColumn(784, 1);

	std::vector<int> trainPredictions;
	trainPredictions.reserve(input_T.getCols());
	
	std::cout << get_time() << ": Starting training..." << std::endl;
	for (int epoch = 0; epoch < epochs; epoch++) {
		float epochLoss = 0.0f;
		int correct = 0;
		int totalSamples = 0;

		for (int start = 0; start < input_T.getCols(); start += batchSize) {
			int end = std::min(start + batchSize, static_cast<int>(input_T.getCols()));
			int currentBatchSize = end - start;

			// Use either standard size Matrix, or the smaller one if processing the last partial batch
			Matrix *inputBatch;
			Matrix *outputBatch;
			if (currentBatchSize == batchSize) {
				inputBatch = &inputBatchStandard;
				outputBatch = &outputBatchStandard;
			} else {
				inputBatch = &inputBatchRest;
				outputBatch = &outputBatchRest;
			}

			for (int i = 0; i < currentBatchSize; i++) {
				input_T.getColumn(start + i, batchColumn);

				// Copy to input batch
				for (int j = 0; j < inputBatch->getRows(); j++) {
					(*inputBatch)(j, i) = batchColumn(j, 0);
				}
				
				// One-hot encode labels
				int label = static_cast<int>(output_T(0, start + i));
				for (int j = 0; j < 10; j++) {
					(*outputBatch)(j, i) = (j == label) ? 1.0f : 0.0f;
				}
			}

			// Forward pass
			const Matrix& predictions = *network.forward(*inputBatch);

			// Compute loss
			epochLoss += softMaxCrossEntropyLoss(predictions, *outputBatch);

			// Compute derivative of loss
			Matrix dLoss(predictions.getRows(), predictions.getCols());
			softMaxCrossEntropyLoss_derivative(dLoss, predictions, *outputBatch);

			// Backward pass
			network.backward(dLoss);
			network.updateAdam(learningRate);

			// Calculate accuracy
			for (int i = 0; i < currentBatchSize; i++) {
				int predictedClass = 0;
				float maxProb = predictions(0, i);
				for (int j = 1; j < predictions.getRows(); j++) {
					if (predictions(j, i) > maxProb) {
						maxProb = predictions(j, i);
						predictedClass = j;
					}
				}

				if (epoch == epochs - 1) {
					trainPredictions.push_back(predictedClass);
				}

				int trueClass = static_cast<int>(output_T(0, start + i));
				if (predictedClass == trueClass) {
					correct++;
				}
				totalSamples++;
			}
		}

		float accuracy = static_cast<float>(correct) / totalSamples * 100.0f;
		std::cout << get_time() << ": Epoch: " << epoch + 1
			<< ", Avg. Loss: " << (epochLoss / (input_T.getCols() / batchSize))
			<< ", Accuracy: " << accuracy << "%" << std::endl;
	}

	std::cout << get_time() << " Training finished." << std::endl;

	dumpCSV(trainDumpFile , trainPredictions);
	std::cout << "Saved Train Predictions to train_predictions.csv" << std::endl;

	std::cout << get_time() << " Test Data Validation." << std::endl;

	int correct = 0;
	int totalSamples = 0;

	std::vector<int> testPredictions;
	testPredictions.reserve(testInput_T.getCols());

	for (int start = 0; start < testInput_T.getCols(); start += batchSize) {
		int end = std::min(start + batchSize, static_cast<int>(testInput_T.getCols()));
		int currentBatchSize = end - start;

		Matrix *inputBatch;
		Matrix *outputBatch;
		if (currentBatchSize == batchSize) {
			inputBatch = &inputBatchStandard;
			outputBatch = &outputBatchStandard;
		} else {
			inputBatch = &inputBatchRestTest;
			outputBatch = &outputBatchRestTest;
		}

		for (int i = 0; i < currentBatchSize; i++) {
			Matrix tempColumn(784, 1);
			testInput_T.getColumn(start + i, tempColumn);

			// Copy to input batch
			for (int j = 0; j < inputBatch->getRows(); j++) {
				(*inputBatch)(j, i) = tempColumn(j, 0);
			}

			// One-hot encode labels
			int label = static_cast<int>(testOutput_T(0, start + i));
			for (int j = 0; j < 10; j++) {
				(*outputBatch)(j, i) = (j == label) ? 1.0f : 0.0f;
			}
		}

		// Forward pass
		const Matrix& predictions = *network.forward(*inputBatch);

		// Calculate accuracy
		for (int i = 0; i < currentBatchSize; i++) {
			int predictedClass = 0;
			float maxProb = predictions(0, i);
			for (int j = 1; j < predictions.getRows(); j++) {
				if (predictions(j, i) > maxProb) {
					maxProb = predictions(j, i);
					predictedClass = j;
				}
			}

			testPredictions.push_back(predictedClass);

			int trueClass = static_cast<int>(testOutput_T(0, start + i));
			if (predictedClass == trueClass) {
				correct++;
			}
			totalSamples++;
		}
	}

	float accuracy = static_cast<float>(correct) / totalSamples * 100.0f;
	std::cout << get_time() << " Accuracy: " << accuracy << "%" << std::endl;

	dumpCSV(testDumpFile, testPredictions);
	std::cout << "Saved Test Predictions to test_predictions.csv" << std::endl;
}

#pragma region XOR
void TestXOR() {
	constexpr int EPOCHS = 500;
	constexpr float LEARNING_RATE = 0.1f;	// Good for Adam
	//constexpr float LEARNING_RATE = 5.0f; // Very good for SGD

	// Makes learning much faster on XOR
	constexpr float WEIGHT_INIT_MIN = -2.0f;
	constexpr float WEIGHT_INIT_MAX = 2.0f;

	const csvFile inputsData = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	const csvFile outputsData = {
		{0},
		{1},
		{1},
		{0}
	};

	// Transform data
	Matrix input = Matrix::fromCSV(inputsData); // 4×2
	Matrix input_T = Matrix::transpose(input); // 2×4
	Matrix output = Matrix::fromCSV(outputsData);	// 4×1
	Matrix output_T = Matrix::transpose(output); // 1×4

	// Set up the network
	Network network;

	Layer hiddenLayer(2, 8, sigmoid, sigmoid_derivative, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
	Layer outputLayer(8, 1, sigmoid, sigmoid_derivative, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);

	network.addLayer(hiddenLayer);
	network.addLayer(outputLayer);

	// Pre-allocate stuff
	Matrix inputColumn(2, 1);
	Matrix outputColumn(1, 1);
	Matrix dLoss(1, 1);

	// Training loop
	for (int epoch = 0; epoch <= EPOCHS; epoch++) {
		float epochLoss = 0.0f;

		// Loop over each sample
		for (int i = 0; i < input_T.getCols(); i++) {
			// Get a single row of input and output as 1×n matrix
			input_T.getColumn(i, inputColumn); // 2×1
			output_T.getColumn(i, outputColumn); // 1×1

			// Forward pass
			const Matrix *prediction = network.forward(inputColumn);

			// Compute loss for logging
			epochLoss += meanSquaredError(*prediction, outputColumn);

			// Compute derivative of MSE
			meanSquaredError_derivative(dLoss, *prediction, outputColumn);

			// Backward pass
			network.backward(dLoss);

			// Update weights
			network.updateAdam(LEARNING_RATE);
		}

		if (epoch % 100 == 0) {
			std::cout << "Epoch: " << epoch << ", AVG loss: " << (epochLoss / input_T.getCols()) << std::endl;
		}
	}

	std::cout << "Learning finished..." << std::endl << std::endl;

	// Test predictions
	for (int i = 0; i < input_T.getCols(); i++) {
		input_T.getColumn(i, inputColumn);
		const Matrix* prediction = network.forward(inputColumn);

		std::cout << "Input: " << inputColumn(0, 0) << " " << inputColumn(1, 0)
			<< " => Prediction: " << (*prediction)(0, 0) << std::endl;
	}
}
#pragma endregion


int main(int argc, char* argv[])
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << "<epochs> <learning_rate> <batch_size>" << std::endl;
		return 1;
	}

	// Parse epochs (int)
	errno = 0;
	char* end1;
	long epochs_long = strtol(argv[1], &end1, 10);
	if (errno != 0 || *end1 != '\0' || epochs_long <= 0) {
		std::cerr << "Invalid epochs: " << argv[1] << std::endl;
		return 1;
	}
	int epochs = static_cast<int>(epochs_long);

	// Parse learning rate (float)
	errno = 0;
	char* end2;
	float learning_rate = std::strtof(argv[2], &end2);
	if (errno != 0 || *end2 != '\0' || learning_rate <= 0.0) {
		std::cerr << "Invalid learning rate: " << argv[2] << "\n";
		return 1;
	}

	// Parse batch size (int)
	errno = 0;
	char* end3;
	long batch_long = std::strtol(argv[3], &end3, 10);
	if (errno != 0 || *end3 != '\0' || batch_long <= 0) {
		std::cerr << "Invalid batch size: " << argv[3] << "\n";
		return 1;
	}
	int batch_size = static_cast<int>(batch_long);
	
	TestFMNIST(epochs, learning_rate, batch_size);
	return 0;
}
