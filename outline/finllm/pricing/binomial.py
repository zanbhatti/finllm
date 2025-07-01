from finllm.pricing.base import Option
from math import log, sqrt, exp
from scipy.stats import norm

class AmericanOption(Option):
    def __init__(self, expiry, strike, option_type, volatility, rate, dividend_yield=0, steps=100):
        super().__init__(expiry, strike, option_type, volatility, rate, dividend_yield)
        self.steps = steps

    def price(self, spot_price):
        return self.binomial_tree(spot_price)
    

    def binomial_tree(self, spot_price, steps=100):

        steps = self.steps

        dt = self.expiry / steps
        u = exp(self.volatility * sqrt(dt))
        d = 1 / u
        disc = exp(-self.rate * dt)
        p = (exp((self.rate - self.dividend_yield) * dt) - d) / (u - d)
        
        option_values = []
        for i in range(steps + 1):
            ST = spot_price * (u ** (steps - i)) * (d ** i)
            if self.option_type == 'call':
                option_values.append(max(0, ST - self.strike))
            else:
                option_values.append(max(0, self.strike - ST))

        for step in range(steps -1, -1, -1):
            for i in range(step + 1):
                ST = spot_price * (u ** (step - i)) * (d ** i)
                
                continuation_val = disc * (p * option_values[i] + (1 - p) * option_values[i + 1])
                if self.option_type == 'call':
                    exercise = max(0, ST - self.strike)
                else:
                    exercise = max(0, self.strike - ST)

                option_values[i] = max(continuation_val, exercise)

        return option_values[0]