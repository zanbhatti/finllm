from finllm.pricing.base import Option
from math import sqrt, exp

class BermudanOption(Option):
    def __init__(self, expiry, strike, option_type, volatility, rate,
                 dividend_yield=0, steps=100, exercise_dates=None):
        super().__init__(expiry, strike, option_type, volatility, rate, dividend_yield)
        self.steps = steps
        if exercise_dates is None:
            # If no exercise dates are provided, it is a European option.
            self.exercise_dates = [expiry]
        else:
            self.exercise_dates = exercise_dates

    def price(self, spot_price):
        return self._binomial_tree(spot_price)

    from math import sqrt, exp

    def _binomial_tree(self, spot_price):
        N = self.steps
        dt = self.expiry / N

        # CRR params (QuantLib-style: continuous compounding)
        u = exp(self.volatility * sqrt(dt))
        d = 1.0 / u
        disc = exp(-self.rate * dt)
        p = (exp((self.rate - self.dividend_yield) * dt) - d) / (u - d)

        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid risk-neutral probability p={p:.6f}")

        # Map exercise times -> step indices
        # - If a time maps to N (maturity), mark terminal exercise allowed.
        # - Otherwise, collect steps strictly between 0 and N.
        exercise_steps = set()
        allow_terminal_exercise = False
        tol = 1e-12
        for t in self.exercise_dates:
            k_float = t / dt
            k = int(round(k_float))
            if abs(k - N) <= 0:  # maps to maturity
                allow_terminal_exercise = True
            elif 1 <= k <= N - 1:
                exercise_steps.add(k)

        K = self.strike
        is_call = (self.option_type == 'call')

        # Build stock price lattice explicitly to align with QL node layout
        stock = [[0.0] * (i + 1) for i in range(N + 1)]
        stock[0][0] = spot_price
        for i in range(1, N + 1):
            stock[i][0] = stock[i - 1][0] * u
            for j in range(1, i + 1):
                stock[i][j] = stock[i - 1][j - 1] * d

        # Terminal layer:
        # - If maturity is an allowed exercise date, payoff as usual.
        # - Else, zero (cannot exercise at T in a QL Bermudan unless included).
        values = [0.0] * (N + 1)
        if allow_terminal_exercise:
            for j in range(N + 1):
                intrinsic = (stock[N][j] - K) if is_call else (K - stock[N][j])
                values[j] = intrinsic if intrinsic > 0.0 else 0.0
        else:
            # No exercise at maturity allowed -> terminal value is zero everywhere.
            for j in range(N + 1):
                values[j] = 0.0

        # Backward induction with Bermudan exercise only at specified steps
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                continuation = disc * (p * values[j] + (1.0 - p) * values[j + 1])
                if i in exercise_steps:
                    intrinsic = (stock[i][j] - K) if is_call else (K - stock[i][j])
                    if intrinsic < 0.0:
                        intrinsic = 0.0
                    values[j] = max(continuation, intrinsic)
                else:
                    values[j] = continuation

        return values[0]

