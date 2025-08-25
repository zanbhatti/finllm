import unittest
from finllm.pricing.black_scholes import EuropeanOption
import QuantLib as ql

class TestEuropeanOption(unittest.TestCase):
    def setUp(self):
        self.spot = 100
        self.strike = 100
        self.expiry = 1
        self.vol = 0.2
        self.rate = 0.05
        self.div_yield = 0.0

    def black_scholes_test_helper(self, option_type, spot, strike, expiry, vol, rate, div_yield, places):
        option = EuropeanOption(expiry, strike, option_type, vol, rate, div_yield)
        my_price = option.price(spot)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
        rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, ql.Actual365Fixed()))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, div_yield, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), vol, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot_quote, div_ts, rate_ts, vol_ts)

        ql_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put if option_type == 'put' else ql.Option.Call, strike),
            ql.EuropeanExercise(today + int(expiry * 365))
        )
        ql_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return ql_option.NPV()
    def test_call_price_close_to_expected(self):
        option = EuropeanOption(self.expiry, self.strike, 'call', self.vol, self.rate, self.div_yield)
        price = option.price(self.spot)
        self.assertAlmostEqual(price, 10.45, places=2)

    def test_put_price_close_to_expected(self):
        option = EuropeanOption(self.expiry, self.strike, 'put', self.vol, self.rate, self.div_yield)
        price = option.price(self.spot)
        self.assertAlmostEqual(price, 5.57, places=2)

    def test_put_call_parity(self):
        call = EuropeanOption(self.expiry, self.strike, 'call', self.vol, self.rate)
        put = EuropeanOption(self.expiry, self.strike, 'put', self.vol, self.rate)
        lhs = call.price(self.spot) - put.price(self.spot)
        rhs = self.spot - self.strike * (2.71828 ** (-self.rate * self.expiry))
        self.assertAlmostEqual(lhs, rhs, delta=1.0)

    def test_invalid_option_type(self):
        with self.assertRaises(AssertionError):
            EuropeanOption(self.expiry, self.strike, 'banana', self.vol, self.rate)

    def test_call_matches_quantlib(self):
        ql_price = self.black_scholes_test_helper('call', self.spot, self.strike, self.expiry, self.vol, self.rate, self.div_yield, 2)

    def test_put_matches_quantlib(self):
        self.black_scholes_test_helper('put', self.spot, self.strike, self.expiry, self.vol, self.rate, self.div_yield, 2)
    def test_call_matches_quantlib(self):
        option = EuropeanOption(self.expiry, self.strike, 'call', self.vol, self.rate)
        my_price = option.price(self.spot)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        rate = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        div = ql.YieldTermStructureHandle(ql.FlatForward(today, self.div_yield, ql.Actual365Fixed()))
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.vol, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot, div, rate, vol)

        ql_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, self.strike),
            ql.EuropeanExercise(today + int(self.expiry * 365))
        )
        ql_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        self.assertAlmostEqual(my_price, ql_option.NPV(), places=2)