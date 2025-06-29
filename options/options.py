from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import streamlit as st
from abc import ABC, abstractmethod

class Option(ABC):
    def __init__(self, expiry, strike, option_type, volatility, rate, underlying_price, dividend_yield=0):
        self.expiry = expiry
        self.strike = strike
        self.option_type = option_type
        self.volatility = volatility
        self.rate = rate
        self.underlying_price = underlying_price
        self.dividend_yield = dividend_yield
        self.created_at = datetime.now()

        # validation
        assert self.option_type in ['call', 'put'], "Option type must be 'call' or 'put'"
        assert self.expiry > 0, "Expiry must be positive"
        assert self.strike > 0, "Strike must be positive"
        assert self.volatility > 0, "Volatility must be positive"
        assert self.rate >= 0, "Interest rate must be non-negative"
        assert self.dividend_yield >= 0, "Dividend yield must be non-negative"

    def black_scholes(self, underlying_price):
        from math import log, sqrt, exp
        from scipy.stats import norm

        d1 = (log(underlying_price / self.strike) + (self.rate + 0.5 * self.volatility ** 2) * self.expiry) / (self.volatility * sqrt(self.expiry))
        d2 = d1 - self.volatility * sqrt(self.expiry)

        if self.option_type == 'call':
            price = (underlying_price * norm.cdf(d1) - self.strike * exp(-self.rate * self.expiry) * norm.cdf(d2))
        elif self.option_type == 'put':
            price = (self.strike * exp(-self.rate * self.expiry) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1))
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return price