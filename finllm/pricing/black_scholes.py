from finllm.pricing.base import Option

class EuropeanOption(Option):
    def price(self, spot_price):
        return self.black_scholes(spot_price)
    

    def black_scholes(self, spot_price):
        from math import log, sqrt, exp
        from scipy.stats import norm

        d1 = (log(spot_price / self.strike) + (self.rate + 0.5 * self.volatility ** 2) * self.expiry) / (self.volatility * sqrt(self.expiry))
        d2 = d1 - self.volatility * sqrt(self.expiry)

        if self.option_type == 'call':
            return spot_price * norm.cdf(d1) - self.strike * exp(-self.rate * self.expiry) * norm.cdf(d2)
        elif self.option_type == 'put':
            return self.strike * exp(-self.rate * self.expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")