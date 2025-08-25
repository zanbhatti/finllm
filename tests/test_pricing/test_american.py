import unittest
from finllm.pricing.black_scholes import EuropeanOption
from finllm.pricing.american import AmericanOption
import QuantLib as ql

class TestAmericanOption(unittest.TestCase):
    def setUp(self):
        self.spot = 100
        self.strike = 100
        self.expiry = 1
        self.vol = 0.2
        self.rate = 0.05
        self.div_yield = 0.0

    def test_american_put_exceeds_european_put(self):
        euro = EuropeanOption(self.expiry, self.strike, 'put', self.vol, self.rate)
        amer = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, steps=200)
        self.assertGreaterEqual(amer.price(self.spot), euro.price(self.spot))

    def test_american_call_equals_european_call(self):
        euro = EuropeanOption(self.expiry, self.strike, 'call', self.vol, self.rate)
        amer = AmericanOption(self.expiry, self.strike, 'call', self.vol, self.rate, steps=1000)
        self.assertAlmostEqual(amer.price(self.spot), euro.price(self.spot), places=2)

    def binomial_test_helper(self, option_type, steps, places):
        my_option = AmericanOption(self.expiry, self.strike, option_type, self.vol, self.rate, steps=steps)
        my_price = my_option.price(self.spot)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        rate = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        div = ql.YieldTermStructureHandle(ql.FlatForward(today, self.div_yield, ql.Actual365Fixed()))
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.vol, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot, div, rate, vol)

        ql_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put if option_type == 'put' else ql.Option.Call, self.strike),
            ql.AmericanExercise(today, today + int(self.expiry * 365))
        )
        ql_option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))

        self.assertAlmostEqual(my_price, ql_option.NPV(), places=places)

    def test_put_matches_quantlib(self):
        self.binomial_test_helper('put', 500, 1)

    def test_call_matches_quantlib(self):
        self.binomial_test_helper('call', 500, 1)

    def test_put_matches_quantlib_steps_100(self):
        self.binomial_test_helper('put', 100, 1)

    def test_call_matches_quantlib_steps_100(self):
        self.binomial_test_helper('call', 100, 1)

    def test_call_with_dividend(self):
        amer = AmericanOption(self.expiry, self.strike, 'call', self.vol, self.rate, dividend_yield=0.03, steps=200)
        self.assertGreater(amer.price(self.spot), 0)

    def test_put_with_dividend(self):
        amer = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, dividend_yield=0.03, steps=200)
        self.assertGreater(amer.price(self.spot), 0)

    def test_different_steps(self):
        price1 = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, steps=50).price(self.spot)
        price2 = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, steps=100).price(self.spot)
        price3 = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, steps=200).price(self.spot)
        self.assertGreaterEqual(price2, price1)
        self.assertGreaterEqual(price3, price2)

    def test_zero_expiry(self):
        with self.assertRaises(AssertionError):
            AmericanOption(0, self.strike, 'put', self.vol, self.rate, steps=200)
