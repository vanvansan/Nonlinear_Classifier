import numpy as np
import matplotlib.pyplot as plt

def phi(z):
    return np.tanh(z)  # You can replace this with your activation function

def phi_prime(z):
    return 1.0 - np.tanh(z)**2  # Derivative of the activation function

def fw(x, w):
    return w[0]*phi(w[1] * x[:, 0] + w[2] * x[:, 1] + w[3] * x[:, 2] + w[4]) + \
           w[5]*phi(w[6] * x[:, 0] + w[7] * x[:, 1] + w[8] * x[:, 2] + w[9]) + \
           w[10]*phi(w[11] * x[:, 0] + w[12] * x[:, 1] + w[13] * x[:, 2] + w[14]) + w[15]

def residual(x, y, w):
    return fw(x, w) - y

def jacobian(x, w):
    phi_prime_1 = phi_prime(w[1] * x[:, 0] + w[2] * x[:, 1] + w[3] * x[:, 2] + w[4])
    phi_prime_2 = phi_prime(w[6] * x[:, 0] + w[7] * x[:, 1] + w[8] * x[:, 2] + w[9])
    phi_prime_3 = phi_prime(w[11] * x[:, 0] + w[12] * x[:, 1] + w[13] * x[:, 2] + w[14])

    jacobian_matrix = np.column_stack([
        phi(w[1] * x[:, 0] + w[2] * x[:, 1] + w[3] * x[:, 2] + w[4]),
        w[0] * x[:, 0] * phi_prime_1,
        w[0] * x[:, 1] * phi_prime_1,
        w[0] * x[:, 2] * phi_prime_1,
        w[0] * phi_prime_1,

        phi(w[6] * x[:, 0] + w[7] * x[:, 1] + w[8] * x[:, 2] + w[9]),
        w[5] * x[:, 0] * phi_prime_2,
        w[5] * x[:, 1] * phi_prime_2,
        w[5] * x[:, 2] * phi_prime_2,
        w[5] * phi_prime_2,

        phi(w[11] * x[:, 0] + w[12] * x[:, 1] + w[13] * x[:, 2] + w[14]),
        w[10] * x[:, 0] * phi_prime_3,
        w[10] * x[:, 1] * phi_prime_3,
        w[10] * x[:, 2] * phi_prime_3,
        w[10] * phi_prime_3,

        np.ones_like(y)
    ])

    return jacobian_matrix

def levenberg_marquardt(x, y, w_init, lambda_val=0.00005, max_iterations=100, tol=1e-6):
    w = w_init.copy()
    iterations = 0
    losses = []

    while iterations < max_iterations:
        r = residual(x, y, w)
        J = jacobian(x, w)

        A = J.T @ J + lambda_val * np.eye(len(w))
        b = J.T @ r

        delta_w = np.linalg.solve(A, -b)
        w_new = w + delta_w

        loss = np.sum(r**2)
        losses.append(loss)

        if np.linalg.norm(delta_w) < tol:
            break

        if np.sum(residual(x, y, w_new)**2) < loss:
            lambda_val /= 10
            w = w_new
        else:
            lambda_val *= 10

        iterations += 1

    return w, losses

# Generate random data
np.random.seed(42)
N = 500
x = np.random.rand(N, 3) * 2 - 1
y = x[:, 0] * x[:, 1] + x[:, 2]

# Initialize weights
w_init = np.random.rand(16)

# Run Levenberg-Marquardt algorithm
learned_w, loss_history = levenberg_marquardt(x, y, w_init, lambda_val=0.00005)

# Plot training loss versus iterations
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Iterations')
plt.show()

print("Learned weights:")
print(learned_w)