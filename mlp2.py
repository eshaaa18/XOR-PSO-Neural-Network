import numpy as np
import matplotlib.pyplot as plt


class MLP:
    """
    Multi-Layer Perceptron (MLP) for XOR problem.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes weights, biases, and other parameters.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def _sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Performs the forward pass through the network.
        """
        # Hidden layer
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = self._sigmoid(self.hidden_input)

        # Output layer
        self.output_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.output = self._sigmoid(self.output_input)

        return self.output

    def backward(self, X, y, output):
        """
        Implements backpropagation to compute gradients.
        """
        # Calculate loss and its derivative
        self.error = y - output
        self.d_loss = self.error  # Based on MSE formula L = 1/2 * (y - y_hat)^2

        # Gradients for output layer
        d_output = self.d_loss * self._sigmoid_derivative(output)
        self.dW2 = np.dot(self.hidden_output.T, d_output)
        self.db2 = np.sum(d_output, axis=0, keepdims=True)

        # Gradients for hidden layer
        d_hidden = np.dot(d_output, self.W2.T) * self._sigmoid_derivative(self.hidden_output)
        self.dW1 = np.dot(X.T, d_hidden)
        self.db1 = np.sum(d_hidden, axis=0, keepdims=True)

    def update_weights(self):
        """
        Updates weights using gradient descent.
        """
        self.W1 += self.learning_rate * self.dW1
        self.b1 += self.learning_rate * self.db1
        self.W2 += self.learning_rate * self.dW2
        self.b2 += self.learning_rate * self.db2

    def train(self, X, y, epochs):
        """
        Trains the network for a specified number of epochs.
        """
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            self.update_weights()

            loss = 0.5 * np.sum((y - output) ** 2)
            losses.append(loss)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

            if loss <= 0.01:
                print(f"Converged at epoch {epoch + 1} with loss: {loss:.4f}")
                return losses, epoch + 1

        return losses, epochs

    def predict(self, X):
        """
        Returns predictions for a given input.
        """
        return self.forward(X)


# Main script to run the models and compare them
if __name__ == "__main__":
    # XOR truth table data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # --- Model 1: 2 hidden neurons ---
    print("--- Training Model 1 (2 hidden neurons) ---")
    mlp1 = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    losses1, epochs_converged1 = mlp1.train(X_train, y_train, epochs=100000)

    # --- Model 2: 4 hidden neurons ---
    print("\n--- Training Model 2 (4 hidden neurons) ---")
    mlp2 = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    losses2, epochs_converged2 = mlp2.train(X_train, y_train, epochs=100000)

    # --- Print Final Results ---
    print("\n--- Final Results ---")

    # Model 1 results
    print("\nModel 1 (2 hidden neurons):")
    final_loss1 = losses1[-1] if losses1 else 0
    print(f"Final Loss: {final_loss1:.4f}")
    predictions1 = mlp1.predict(X_train)
    print("Predicted Outputs:")
    for i in range(len(X_train)):
        print(
            f"Input: {X_train[i]} -> Predicted: {predictions1[i][0]:.4f} (Threshold 0.5: {1 if predictions1[i][0] > 0.5 else 0})")
    print(f"Epochs to converge (error <= 0.01): {epochs_converged1}")

    # Model 2 results
    print("\nModel 2 (4 hidden neurons):")
    final_loss2 = losses2[-1] if losses2 else 0
    print(f"Final Loss: {final_loss2:.4f}")
    predictions2 = mlp2.predict(X_train)
    print("Predicted Outputs:")
    for i in range(len(X_train)):
        print(
            f"Input: {X_train[i]} -> Predicted: {predictions2[i][0]:.4f} (Threshold 0.5: {1 if predictions2[i][0] > 0.5 else 0})")
    print(f"Epochs to converge (error <= 0.01): {epochs_converged2}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Model 1 plot
    plt.plot(losses1, label=f"Model 1 (2 Hidden Neurons) - Converged at {epochs_converged1} epochs")

    # Model 2 plot
    plt.plot(losses2, label=f"Model 2 (4 Hidden Neurons) - Converged at {epochs_converged2} epochs")

    plt.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold (Loss = 0.01)')
    plt.title("Error Convergence Curves for MLP Models on XOR Problem")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.5)
    plt.show()