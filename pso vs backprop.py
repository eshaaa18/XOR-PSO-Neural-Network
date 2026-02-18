# Title : Solving XOR Problem using PSO and MLP (Backpropagation)
# Author: Sneha Tiwary & Jahnavi Srivastava
# Year  : 2025-2026
# -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Activation Function ----------------
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------------- Fitness Function ----------------
def fitness(params, X, y, input_size, hidden_size, output_size):
    idx = 0
    W1 = params[idx:idx + input_size * hidden_size].reshape(input_size, hidden_size)
    idx += input_size * hidden_size
    b1 = params[idx:idx + hidden_size].reshape(1, hidden_size)
    idx += hidden_size
    W2 = params[idx:idx + hidden_size * output_size].reshape(hidden_size, output_size)
    idx += hidden_size * output_size
    b2 = params[idx:idx + output_size].reshape(1, output_size)

    # Forward Pass
    H = sigmoid(np.dot(X, W1) + b1)
    y_pred = sigmoid(np.dot(H, W2) + b2)
    return np.mean((y - y_pred) ** 2)


# ===============================================================
#                PSO MLP Class
# ===============================================================
class PSO_MLP:
    def __init__(self, input_size, hidden_size, output_size, n_particles=30, max_iter=300):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dim = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = 0.729, 1.494, 1.494
        self.losses = []
        self.best_epoch = 0
        self.best_loss = np.inf

    def train(self, X, y):
        pos = np.random.uniform(-1, 1, (self.n_particles, self.dim))
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        pbest_val = np.array([fitness(p, X, y, self.input_size, self.hidden_size, self.output_size) for p in pos])
        gbest = pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        self.losses = [gbest_val]
        self.best_loss = gbest_val
        self.best_epoch = 1

        for t in range(self.max_iter):
            for i in range(self.n_particles):
                val = fitness(pos[i], X, y, self.input_size, self.hidden_size, self.output_size)
                if val < pbest_val[i]:
                    pbest[i], pbest_val[i] = pos[i].copy(), val
            if np.min(pbest_val) < gbest_val:
                gbest = pbest[np.argmin(pbest_val)].copy()
                gbest_val = np.min(pbest_val)
                self.best_epoch = t + 1
                self.best_loss = gbest_val
            self.losses.append(gbest_val)

            # Velocity & Position update
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = (self.w * vel[i]
                          + self.c1 * r1 * (pbest[i] - pos[i])
                          + self.c2 * r2 * (gbest - pos[i]))
                pos[i] += vel[i]

            if gbest_val < 1e-6:
                break

        self.best_particle = gbest

    def predict(self, X):
        p = self.best_particle
        idx = 0
        W1 = p[idx:idx + self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        idx += self.input_size * self.hidden_size
        b1 = p[idx:idx + self.hidden_size].reshape(1, self.hidden_size)
        idx += self.hidden_size
        W2 = p[idx:idx + self.hidden_size * self.output_size].reshape(self.hidden_size, self.output_size)
        idx += self.hidden_size * self.output_size
        b2 = p[idx:idx + self.output_size].reshape(1, self.output_size)
        H = sigmoid(np.dot(X, W1) + b1)
        y_pred = sigmoid(np.dot(H, W2) + b2)
        return y_pred


# ===============================================================
#                Standard MLP (Backpropagation)
# ===============================================================
class MLP_Backprop:
    def __init__(self, input_size, hidden_size, output_size, lr=0.5, epochs=3000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs
        self.losses = []
        self.best_epoch = 0
        self.best_loss = np.inf

        # Initialize weights
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward
            H = sigmoid(np.dot(X, self.W1) + self.b1)
            y_pred = sigmoid(np.dot(H, self.W2) + self.b2)

            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

            # Update best
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = epoch + 1

            # Backpropagation
            d_output = (y_pred - y) * sigmoid_derivative(y_pred)
            d_hidden = np.dot(d_output, self.W2.T) * sigmoid_derivative(H)

            # Gradient update
            self.W2 -= self.lr * np.dot(H.T, d_output)
            self.b2 -= self.lr * np.sum(d_output, axis=0, keepdims=True)
            self.W1 -= self.lr * np.dot(X.T, d_hidden)
            self.b1 -= self.lr * np.sum(d_hidden, axis=0, keepdims=True)

            if loss < 1e-6:
                break

    def predict(self, X):
        H = sigmoid(np.dot(X, self.W1) + self.b1)
        y_pred = sigmoid(np.dot(H, self.W2) + self.b2)
        return y_pred


# ===============================================================
#                Main XOR Training & Comparison
# ===============================================================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train PSO-based MLP
pso_mlp = PSO_MLP(input_size=2, hidden_size=2, output_size=1, max_iter=300)
pso_mlp.train(X, y)
pso_pred = pso_mlp.predict(X)

# Train Backprop-based MLP
mlp = MLP_Backprop(input_size=2, hidden_size=2, output_size=1, lr=0.5, epochs=3000)
mlp.train(X, y)
mlp_pred = mlp.predict(X)

# Print Predictions
print("\n===== Predictions (PSO_MLP) =====")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {pso_pred[i][0]:.4f}, Target: {y[i][0]}")
print(f"\n⚙️  PSO converged at iteration {pso_mlp.best_epoch} with best MSE = {pso_mlp.best_loss:.8f}")

print("\n===== Predictions (MLP_Backprop) =====")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {mlp_pred[i][0]:.4f}, Target: {y[i][0]}")
print(f"\n⚙️  MLP converged at epoch {mlp.best_epoch} with best MSE = {mlp.best_loss:.8f}")

# ===============================================================
#                Plot Convergence Comparison
# ===============================================================
plt.figure(figsize=(10, 6))
plt.plot(pso_mlp.losses, label='PSO_MLP Loss', linewidth=2)
plt.plot(mlp.losses, label='MLP_Backprop Loss', linewidth=2)

# Mark convergence points
plt.scatter(pso_mlp.best_epoch, pso_mlp.best_loss, color='blue', s=70, marker='o')
plt.scatter(mlp.best_epoch, mlp.best_loss, color='orange', s=70, marker='o')

# Annotate convergence
plt.text(pso_mlp.best_epoch, pso_mlp.best_loss * 1.5,
         f'PSO: {pso_mlp.best_epoch} epochs\nLoss={pso_mlp.best_loss:.6f}', color='blue')
plt.text(mlp.best_epoch, mlp.best_loss * 1.5,
         f'MLP: {mlp.best_epoch} epochs\nLoss={mlp.best_loss:.6f}', color='orange')

plt.title("Convergence Comparison: PSO vs Backpropagation (XOR Problem)")
plt.xlabel("Epochs / Iterations")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()
