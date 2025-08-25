import unittest
from finllm.pricing.black_scholes import EuropeanOption
from finllm.pricing.binomial import AmericanOption
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

    def test_put_matches_quantlib(self):
        my_option = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, steps=500)
        my_price = my_option.price(self.spot)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        rate = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        div = ql.YieldTermStructureHandle(ql.FlatForward(today, self.div_yield, ql.Actual365Fixed()))
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.vol, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot, div, rate, vol)

        ql_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, self.strike),
            ql.AmericanExercise(today, today + int(self.expiry * 365))
        )
        ql_option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 500))

        self.assertAlmostEqual(my_price, ql_option.NPV(), places=1)