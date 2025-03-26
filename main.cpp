#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip> // For formatting output
#include <fstream> // For saving/loading model
#include "random_shuffle.h"

using namespace std;

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
    return max(0.0, x);
}

double leakyRelu(double x) {
    return x > 0 ? x : 0.01 * x; // Leaky ReLU
}

double tanhActivation(double x) {
    return tanh(x);
}

// Derivatives of activation functions
double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double leakyReluDerivative(double x) {
    return x > 0 ? 1.0 : 0.01; // Leaky ReLU derivative
}

double tanhDerivative(double x) {
    return 1.0 - x * x;
}

// Save model to file
void saveModel(const vector<vector<double>>& weightsInputHidden, const vector<vector<double>>& weightsHiddenOutput,
               const vector<double>& biasHidden, const vector<double>& biasOutput, const string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "Error: Could not open file for saving model." << endl;
        return;
    }

    // Save weightsInputHidden
    outFile << "weightsInputHidden:\n";
    for (const auto& row : weightsInputHidden) {
        for (double val : row) outFile << val << " ";
        outFile << "\n";
    }

    // Save weightsHiddenOutput
    outFile << "weightsHiddenOutput:\n";
    for (const auto& row : weightsHiddenOutput) {
        for (double val : row) outFile << val << " ";
        outFile << "\n";
    }

    // Save biasHidden
    outFile << "biasHidden:\n";
    for (double val : biasHidden) outFile << val << " ";
    outFile << "\n";

    // Save biasOutput
    outFile << "biasOutput:\n";
    for (double val : biasOutput) outFile << val << " ";
    outFile << "\n";

    outFile.close();
}

// Load model from file
void loadModel(vector<vector<double>>& weightsInputHidden, vector<vector<double>>& weightsHiddenOutput,
               vector<double>& biasHidden, vector<double>& biasOutput, const string& filename) {
    ifstream inFile(filename);
    if (!inFile) {
        cerr << "Error: Could not open file for loading model." << endl;
        return;
    }

    string section;
    while (inFile >> section) {
        if (section == "weightsInputHidden:") {
            for (auto& row : weightsInputHidden) {
                for (double& val : row) inFile >> val;
            }
        } else if (section == "weightsHiddenOutput:") {
            for (auto& row : weightsHiddenOutput) {
                for (double& val : row) inFile >> val;
            }
        } else if (section == "biasHidden:") {
            for (double& val : biasHidden) inFile >> val;
        } else if (section == "biasOutput:") {
            for (double& val : biasOutput) inFile >> val;
        }
    }

    inFile.close();
}

int main() {
    srand(time(0));

    // Training data for XOR gate (more complex than AND)
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> outputs = {0, 1, 1, 0};

    // Network architecture
    int inputSize = 2;
    int hiddenSize = 4; // Hidden layer size
    int outputSize = 1;

    // Xavier or He initialization for weights
    vector<vector<double>> weightsInputHidden(hiddenSize, vector<double>(inputSize));
    vector<vector<double>> weightsHiddenOutput(outputSize, vector<double>(hiddenSize));
    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weightsInputHidden[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / inputSize); // He initialization
        }
    }
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            weightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / hiddenSize); // He initialization
        }
    }

    // Biases
    vector<double> biasHidden(hiddenSize, 0.0);
    vector<double> biasOutput(outputSize, 0.0);

    // Hyperparameters
    double learningRate = 0.001;
    int epochs = 10000;
    int batchSize = 2; // Mini-batch size
    double dropoutRate = 0.5; // Dropout rate
    double gradientClipThreshold = 5.0; // Gradient clipping threshold
    double warmupEpochs = 1000; // Learning rate warmup epochs
    double epsilon = 1e-8; // Small value for numerical stability
    double beta1 = 0.9; // Adam optimizer parameter
    double beta2 = 0.999; // Adam optimizer parameter

    // Adam optimizer variables
    vector<vector<double>> mWeightsInputHidden(hiddenSize, vector<double>(inputSize, 0.0));
    vector<vector<double>> vWeightsInputHidden(hiddenSize, vector<double>(inputSize, 0.0));
    vector<vector<double>> mWeightsHiddenOutput(outputSize, vector<double>(hiddenSize, 0.0));
    vector<vector<double>> vWeightsHiddenOutput(outputSize, vector<double>(hiddenSize, 0.0));
    vector<double> mBiasHidden(hiddenSize, 0.0), vBiasHidden(hiddenSize, 0.0);
    vector<double> mBiasOutput(outputSize, 0.0), vBiasOutput(outputSize, 0.0);

    // Training loop
    double bestError = numeric_limits<double>::infinity();
    int noImprovementCount = 0;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        double totalError = 0.0;

        // Shuffle data for mini-batch training
        vector<int> indices(inputs.size());
        for (int i = 0; i < indices.size(); i++) indices[i] = i;


        for (int batchStart = 0; batchStart < indices.size(); batchStart += batchSize) {
            vector<int> batchIndices(indices.begin() + batchStart, indices.begin() + min(batchStart + batchSize, (int)indices.size()));

            vector<vector<double>> deltaWeightsInputHidden(hiddenSize, vector<double>(inputSize, 0.0));
            vector<vector<double>> deltaWeightsHiddenOutput(outputSize, vector<double>(hiddenSize, 0.0));
            vector<double> deltaBiasHidden(hiddenSize, 0.0);
            vector<double> deltaBiasOutput(outputSize, 0.0);

            for (int idx : batchIndices) {
                double x1 = inputs[idx][0];
                double x2 = inputs[idx][1];
                double y = outputs[idx];

                // Forward pass
                vector<double> hiddenLayer(hiddenSize, 0.0);
                for (int i = 0; i < hiddenSize; i++) {
                    hiddenLayer[i] = leakyRelu(weightsInputHidden[i][0] * x1 + weightsInputHidden[i][1] * x2 + biasHidden[i]);
                    // Apply dropout
                    if (((double)rand() / RAND_MAX) < dropoutRate) hiddenLayer[i] = 0.0;
                }

                double output = sigmoid(weightsHiddenOutput[0][0] * hiddenLayer[0] +
                                        weightsHiddenOutput[0][1] * hiddenLayer[1] +
                                        weightsHiddenOutput[0][2] * hiddenLayer[2] +
                                        weightsHiddenOutput[0][3] * hiddenLayer[3] + biasOutput[0]);

                // Compute error (MSE loss)
                double error = y - output;
                totalError += error * error;

                // Backpropagation
                double deltaOutput = error * sigmoidDerivative(output);
                vector<double> deltaHidden(hiddenSize, 0.0);
                for (int i = 0; i < hiddenSize; i++) {
                    deltaHidden[i] = deltaOutput * weightsHiddenOutput[0][i] * leakyReluDerivative(hiddenLayer[i]);
                }

                // Update deltas
                for (int i = 0; i < hiddenSize; i++) {
                    deltaWeightsInputHidden[i][0] += deltaHidden[i] * x1;
                    deltaWeightsInputHidden[i][1] += deltaHidden[i] * x2;
                    deltaBiasHidden[i] += deltaHidden[i];
                }
                for (int i = 0; i < hiddenSize; i++) {
                    deltaWeightsHiddenOutput[0][i] += deltaOutput * hiddenLayer[i];
                }
                deltaBiasOutput[0] += deltaOutput;
            }

            // Gradient clipping
            for (auto& row : deltaWeightsInputHidden) {
                for (double& val : row) val = max(-gradientClipThreshold, min(gradientClipThreshold, val));
            }
            for (auto& row : deltaWeightsHiddenOutput) {
                for (double& val : row) val = max(-gradientClipThreshold, min(gradientClipThreshold, val));
            }

            // Adam optimizer updates
            double t = epoch; // Time step
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    mWeightsInputHidden[i][j] = beta1 * mWeightsInputHidden[i][j] + (1 - beta1) * deltaWeightsInputHidden[i][j];
                    vWeightsInputHidden[i][j] = beta2 * vWeightsInputHidden[i][j] + (1 - beta2) * pow(deltaWeightsInputHidden[i][j], 2);
                    double mHat = mWeightsInputHidden[i][j] / (1 - pow(beta1, t));
                    double vHat = vWeightsInputHidden[i][j] / (1 - pow(beta2, t));
                    weightsInputHidden[i][j] += learningRate * mHat / (sqrt(vHat) + epsilon);
                }
                mBiasHidden[i] = beta1 * mBiasHidden[i] + (1 - beta1) * deltaBiasHidden[i];
                vBiasHidden[i] = beta2 * vBiasHidden[i] + (1 - beta2) * pow(deltaBiasHidden[i], 2);
                double mHat = mBiasHidden[i] / (1 - pow(beta1, t));
                double vHat = vBiasHidden[i] / (1 - pow(beta2, t));
                biasHidden[i] += learningRate * mHat / (sqrt(vHat) + epsilon);
            }
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    mWeightsHiddenOutput[i][j] = beta1 * mWeightsHiddenOutput[i][j] + (1 - beta1) * deltaWeightsHiddenOutput[i][j];
                    vWeightsHiddenOutput[i][j] = beta2 * vWeightsHiddenOutput[i][j] + (1 - beta2) * pow(deltaWeightsHiddenOutput[i][j], 2);
                    double mHat = mWeightsHiddenOutput[i][j] / (1 - pow(beta1, t));
                    double vHat = vWeightsHiddenOutput[i][j] / (1 - pow(beta2, t));
                    weightsHiddenOutput[i][j] += learningRate * mHat / (sqrt(vHat) + epsilon);
                }
                mBiasOutput[i] = beta1 * mBiasOutput[i] + (1 - beta1) * deltaBiasOutput[i];
                vBiasOutput[i] = beta2 * vBiasOutput[i] + (1 - beta2) * pow(deltaBiasOutput[i], 2);
                double mHat = mBiasOutput[i] / (1 - pow(beta1, t));
                double vHat = vBiasOutput[i] / (1 - pow(beta2, t));
                biasOutput[i] += learningRate * mHat / (sqrt(vHat) + epsilon);
            }
        }

        // Learning rate warmup
        if (epoch <= warmupEpochs) {
            learningRate = (epoch / (double)warmupEpochs) * learningRate;
        }

        // Early stopping
        if (totalError < bestError) {
            bestError = totalError;
            noImprovementCount = 0;
        } else {
            noImprovementCount++;
        }
        if (noImprovementCount > 1000) {
            cout << "Early stopping at epoch " << epoch << endl;
            break;
        }

        // Print progress
        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " Error: " << totalError << endl;
        }

        // Training progress visualization
        if (epoch % 1000 == 0) {
            int progress = (epoch * 50) / epochs; // Scale to 50 characters
            cout << "[";
            for (int i = 0; i < 50; i++) {
                cout << (i < progress ? '=' : ' ');
            }
            cout << "] " << (epoch * 100 / epochs) << "%\n";
        }

        // Checkpointing
        if (epoch % 5000 == 0) {
            saveModel(weightsInputHidden, weightsHiddenOutput, biasHidden, biasOutput, "checkpoint_" + to_string(epoch) + ".txt");
        }
    }

    // Save the trained model
    saveModel(weightsInputHidden, weightsHiddenOutput, biasHidden, biasOutput, "model.txt");

    // Test the trained network
    cout << "\nTrained Neural Network Results:" << endl;
    for (int j = 0; j < inputs.size(); j++) {
        double x1 = inputs[j][0];
        double x2 = inputs[j][1];
        vector<double> hiddenLayer(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayer[i] = leakyRelu(weightsInputHidden[i][0] * x1 + weightsInputHidden[i][1] * x2 + biasHidden[i]);
        }
        double output = sigmoid(weightsHiddenOutput[0][0] * hiddenLayer[0] +
                                weightsHiddenOutput[0][1] * hiddenLayer[1] +
                                weightsHiddenOutput[0][2] * hiddenLayer[2] +
                                weightsHiddenOutput[0][3] * hiddenLayer[3] + biasOutput[0]);
        cout << "Input: (" << x1 << ", " << x2 << ") -> Output: " << output << endl;
    }

    // Load the saved model
    loadModel(weightsInputHidden, weightsHiddenOutput, biasHidden, biasOutput, "model.txt");
    cout << "\nModel loaded successfully!" << endl;

    // Confusion matrix for evaluation
    vector<vector<int>> confusionMatrix(2, vector<int>(2, 0)); // Binary classification
    for (int j = 0; j < inputs.size(); j++) {
        double x1 = inputs[j][0];
        double x2 = inputs[j][1];
        vector<double> hiddenLayer(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayer[i] = leakyRelu(weightsInputHidden[i][0] * x1 + weightsInputHidden[i][1] * x2 + biasHidden[i]);
        }
        double output = sigmoid(weightsHiddenOutput[0][0] * hiddenLayer[0] +
                                weightsHiddenOutput[0][1] * hiddenLayer[1] +
                                weightsHiddenOutput[0][2] * hiddenLayer[2] +
                                weightsHiddenOutput[0][3] * hiddenLayer[3] + biasOutput[0]);
        int predicted = output > 0.5 ? 1 : 0;
        int actual = outputs[j];
        confusionMatrix[actual][predicted]++;
    }

    cout << "\nConfusion Matrix:\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << confusionMatrix[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}