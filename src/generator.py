import numpy as np

class AsianBasketSimulator:
    def __init__(self, n_assets=4):
        self.n_assets = n_assets
        # Default correlation matrix (symmetric, positive definite)
        self.rho = np.array([
            [1.0, 0.5, 0.3, 0.1],
            [0.5, 1.0, 0.2, 0.0],
            [0.3, 0.2, 1.0, 0.4],
            [0.1, 0.0, 0.4, 1.0]
        ])
        # Cholesky decomposition for correlating assets
        self.L = np.linalg.cholesky(self.rho)

    def generate_data(self, n_samples=10000, n_steps=50):
        # Randomize market parameters for training
        # S0 in [90, 110], K in [90, 110], r in [0.01, 0.05], sigma in [0.1, 0.3], T in [0.5, 2.0]
        S0 = np.random.uniform(90, 110, (n_samples, self.n_assets))
        K = np.random.uniform(90, 110, (n_samples, 1))
        r = np.random.uniform(0.01, 0.05, (n_samples, 1))
        sigma = np.random.uniform(0.1, 0.3, (n_samples, self.n_assets))
        T = np.random.uniform(0.5, 2.0, (n_samples, 1))
        
        # Monte Carlo Simulation
        dt = T / n_steps
        prices = np.zeros((n_samples, n_steps + 1, self.n_assets))
        prices[:, 0, :] = S0
        
        # Standard Normal Random Variables
        Z = np.random.normal(0, 1, (n_samples, n_steps, self.n_assets))
        
        # Correlate Z using Cholesky L
        # Z_corr[i, t] = Z[i, t] @ L.T
        Z_corr = Z @ self.L.T
        
        for t in range(n_steps):
            # Geometric Brownian Motion Step
            drift = (r - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z_corr[:, t, :]
            prices[:, t+1, :] = prices[:, t, :] * np.exp(drift + diffusion)
            
        # Asian Payoff: Arithmetic Average of Basket
        basket_paths = np.mean(prices, axis=2) # Average across assets
        avg_prices = np.mean(basket_paths, axis=1, keepdims=True) # Average over time
        
        # Discounted Payoff
        payoffs = np.maximum(avg_prices - K, 0) * np.exp(-r * T)
        
        # Prepare Feature Vector X: [Mean(S0), K, r, Mean(sigma), T]
        # Using means for simplicity in feature vector, or can flatten all.
        # Here we use a simplified feature set for the model.
        X = np.hstack([np.mean(S0, axis=1, keepdims=True), K, r, np.mean(sigma, axis=1, keepdims=True), T])
        Y = payoffs
        
        return X.astype(np.float32), Y.astype(np.float32)