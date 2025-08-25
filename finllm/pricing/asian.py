from finllm.pricing.base import Option
from math import sqrt, exp
import numpy as np

class AsianOption(Option):
    def __init__(self, expiry, strike, option_type, volatility, rate,
                 dividend_yield=0.0, num_simulations=10000, num_steps=100,
                 antithetic=True, seed=None, include_s0=False):
        """
        Arithmetic-average Asian option via Monte Carlo under risk-neutral GBM.

        Monitoring: equally spaced dates over (0, T] by default (exclude S0).
        Set include_s0=True to average over {S0, S_{t1}, ..., S_{tn}}.
        """
        super().__init__(expiry, strike, option_type, volatility, rate, dividend_yield)
        self.num_simulations = int(num_simulations)
        self.num_steps = int(num_steps)
        self.antithetic = bool(antithetic)
        self.seed = seed
        self.include_s0 = bool(include_s0)

    def price(self, spot_price):
        """
        Returns (price, std_error).
        """
        S0 = float(spot_price)
        K  = float(self.strike)
        T  = float(self.expiry)
        r  = float(self.rate)
        q  = float(self.dividend_yield)
        vol = float(self.volatility)
        n  = int(self.num_steps)
        m  = int(self.num_simulations)

        if T <= 0 or vol < 0 or n <= 0 or m <= 0:
            raise ValueError("Invalid parameters for Asian option pricing.")

        np.random.seed(self.seed)

        dt = T / n
        drift = (r - q - 0.5 * vol * vol) * dt
        diffusion = vol * sqrt(dt)

        if self.antithetic:
            m_half = (m + 1) // 2
            Z = np.random.randn(m_half, n)
            Z = np.vstack([Z, -Z])[:m]
        else:
            Z = np.random.randn(m, n)

        # log-GBM path increments; average over monitoring dates (exclude S0 by default)
        log_paths = np.log(S0) + np.cumsum(drift + diffusion * Z, axis=1)
        S_paths = np.exp(log_paths)  # shape (m, n)

        if self.include_s0:
            A = (S_paths.sum(axis=1) + S0) / (n + 1)
        else:
            A = S_paths.mean(axis=1)

        opt_type = str(self.option_type).lower()
        if opt_type == 'call':
            payoffs = np.maximum(A - K, 0.0)
        elif opt_type == 'put':
            payoffs = np.maximum(K - A, 0.0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        disc = exp(-r * T)
        price = disc * payoffs.mean()
        std_error = disc * payoffs.std(ddof=1) / sqrt(m)
        return price, std_error
