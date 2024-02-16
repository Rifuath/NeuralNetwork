#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Define network architecture constants
#define VARIANCE_W 0.5
#define INPUTS 2
#define HIDDEN 3
#define OUTPUTS 1

// Declare weights, biases, and momentum for hidden and output layers
double hidden_weights[HIDDEN][INPUTS];
double hidden_momentum[HIDDEN][INPUTS];
double hidden_bias[HIDDEN];
double output_weights[OUTPUTS][HIDDEN];
double output_momentum[OUTPUTS][HIDDEN];
double output_bias[OUTPUTS];

// Activation function: Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid
double sigmoid_prime(double x) {
    return x * (1 - x);
}

// Activation function: Hyperbolic Tangent (Tanh)
double tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

// Derivative of Tanh
double tanh_prime(double x) {
    return 1 - x * x;
}

// Predict the output for given inputs using the neural network
double predict(double inputs[INPUTS]) {
    double hiddens[HIDDEN];
    // Forward pass through hidden layer
    for (int i = 0; i < HIDDEN; ++i) {
        double hidden = 0;
        for (int j = 0; j < INPUTS; ++j) {
            hidden += hidden_weights[i][j] * inputs[j];
        }
        hiddens[i] = tanh(hidden + hidden_bias[i]);
    }

    double outputs[OUTPUTS];
    // Forward pass through output layer
    for (int i = 0; i < OUTPUTS; ++i) {
        double output = 0;
        for (int j = 0; j < HIDDEN; ++j) {
            output += output_weights[i][j] * hiddens[j];
        }
        outputs[i] = sigmoid(output + output_bias[i]);
    }

    return outputs[0];
}

// Train the neural network with given inputs and targets
void learn(double inputs[INPUTS], double targets[OUTPUTS], double alpha, double lambd) {
    double hiddens[HIDDEN];
    // Forward pass (same as in predict)
    for (int i = 0; i < HIDDEN; ++i) {
        double hidden = 0;
        for (int j = 0; j < INPUTS; ++j) {
            hidden += hidden_weights[i][j] * inputs[j];
        }
        hiddens[i] = tanh(hidden + hidden_bias[i]);
    }

    double outputs[OUTPUTS];
    // Forward pass to output (same as in predict)
    for (int i = 0; i < OUTPUTS; ++i) {
        double output = 0;
        for (int j = 0; j < HIDDEN; ++j) {
            output += output_weights[i][j] * hiddens[j];
        }
        outputs[i] = sigmoid(output + output_bias[i]);
    }

    // Calculate error and derivative of error for output layer
    double errors[OUTPUTS];
    for (int i = 0; i < OUTPUTS; ++i) {
        errors[i] = targets[i] - outputs[i];
    }

    double derrors[OUTPUTS];
    for (int i = 0; i < OUTPUTS; ++i) {
        derrors[i] = errors[i] * sigmoid_prime(outputs[i]);
    }

    // Backpropagate error to hidden layer
    double ds[HIDDEN];
    for (int i = 0; i < HIDDEN; ++i) {
        ds[i] = 0;
        for (int j = 0; j < OUTPUTS; ++j) {
            ds[i] += derrors[j] * output_weights[j][i] * tanh_prime(hiddens[i]);
        }
    }

    // Update weights and biases with momentum for output layer
    for (int i = 0; i < OUTPUTS; ++i) {
        for (int j = 0; j < HIDDEN; ++j) {
            output_momentum[i][j] = lambd * output_momentum[i][j] + alpha * hiddens[j] * derrors[i];
            output_weights[i][j] += output_momentum[i][j];
        }
        output_bias[i] += alpha * derrors[i];
    }

    // Update weights and biases with momentum for hidden layer
    for (int i = 0; i < HIDDEN; ++i) {
        for (int j = 0; j < INPUTS; ++j) {
            hidden_momentum[i][j] = lambd * hidden_momentum[i][j] + alpha * inputs[j] * ds[i];
            hidden_weights[i][j] += hidden_momentum[i][j];
        }
        hidden_bias[i] += alpha * ds[i];
    }
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    // Define XOR input and output pairs
    double inputs[4][INPUTS] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double outputs[4][OUTPUTS] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Initialize weights and biases randomly
    for (int i = 0; i < HIDDEN; ++i) {
        for (int j = 0; j < INPUTS; ++j) {
            hidden_weights[i][j] = (double)rand() / RAND_MAX * 2 * VARIANCE_W - VARIANCE_W;
            hidden_momentum[i][j] = 0;
        }
        hidden_bias[i] = 0;
    }

    for (int i = 0; i < OUTPUTS; ++i) {
        for (int j = 0; j < HIDDEN; ++j) {
            output_weights[i][j] = (double)rand() / RAND_MAX * 2 * VARIANCE_W - VARIANCE_W;
            output_momentum[i][j] = 0;
        }
        output_bias[i] = 0;
    }

    // Training loop
    for (int epoch = 0; epoch < 10000; ++epoch) {
        // Shuffle training data
        int indexes[4] = {0, 1, 2, 3};
        for (int j = 0; j < 4; ++j) {
            int temp = indexes[j];
            int randomIndex = j + rand() % (4 - j);
            indexes[j] = indexes[randomIndex];
            indexes[randomIndex] = temp;

            learn(inputs[indexes[j]], outputs[indexes[j]], 0.2, 0.8);
        }

        // Print mean squared error every 10000 epochs
        if ((epoch + 1) % 10000 == 0) {
            double cost = 0;
            for (int j = 0; j < 4; ++j) {
                double o = predict(inputs[j]);
                cost += pow(outputs[j][0] - o, 2);
            }
            cost /= 4;
            printf("%d mean squared error: %lf\n", epoch + 1, cost);
        }
    }

    // Display predictions
    for (int i = 0; i < 4; ++i) {
        double result = predict(inputs[i]);
        printf("for input [%lf, %lf], expected %lf, predicted %lf, which is %s\n",
               inputs[i][0], inputs[i][1], outputs[i][0], result,
               round(result) == outputs[i][0] ? "correct" : "incorrect");
    }

    return 0;
}
