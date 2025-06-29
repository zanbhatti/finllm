import unittest
from european import EuropeanOption
from american import AmericanOption
import QuantLib as ql

class TestOptionPricing(unittest.TestCase):

    def setUp(self):
        self.spot_price = 100
        self.strike = 100
        self.expiry = 1
        self.volatility = 0.2
        self.rate = 0.05
        self.dividend_yield = 0.0

    def test_european_call_matches_bs(self):
        option = EuropeanOption(self.expiry, self.strike, 'call', self.volatility, self.rate, self.dividend_yield)
        price = option.price(self.spot_price)
        self.assertAlmostEqual(price, 10.45, places=2)

    def test_european_put_matches_bs(self):
        option = EuropeanOption(self.expiry, self.strike, 'put', self.volatility, self.rate, self.dividend_yield)
        price = option.price(self.spot_price)
        self.assertAlmostEqual(price, 5.57, places=2)

    def test_american_put_greater_than_european(self):
        euro_put = EuropeanOption(self.expiry, self.strike, 'put', self.volatility, self.rate)
        amer_put = AmericanOption(self.expiry, self.strike, 'put', self.volatility, self.rate, steps=200)
        euro_price = euro_put.price(self.spot_price)
        amer_price = amer_put.price(self.spot_price)
        self.assertGreaterEqual(amer_price, euro_price)

    def test_invalid_option_type_raises(self):
        with self.assertRaises(AssertionError):
            EuropeanOption(self.expiry, self.strike, 'banana', self.volatility, self.rate)

    def test_european_put_call_parity(self):
        call = EuropeanOption(self.expiry, self.strike, 'call', self.volatility, self.rate)
        put = EuropeanOption(self.expiry, self.strike, 'put', self.volatility, self.rate)
        call_price = call.price(self.spot_price)
        put_price = put.price(self.spot_price)
        lhs = call_price - put_price
        rhs = self.spot_price - self.strike * (2.71828 ** (-self.rate * self.expiry))
        self.assertAlmostEqual(lhs, rhs, delta=1.0)

    def test_european_call_matches_quantlib(self):
        my_option = EuropeanOption(self.expiry, self.strike, 'call', self.volatility, self.rate)
        my_price = my_option.price(self.spot_price)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.dividend_yield, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.volatility, ql.Actual365Fixed()))
        bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)

        european_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, self.strike),
            ql.EuropeanExercise(today + int(self.expiry * 365))
        )
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

        ql_price = european_option.NPV()
        self.assertAlmostEqual(my_price, ql_price, places=2)

    def test_european_put_matches_quantlib(self):
        my_option = EuropeanOption(self.expiry, self.strike, 'put', self.volatility, self.rate)
        my_price = my_option.price(self.spot_price)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.dividend_yield, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.volatility, ql.Actual365Fixed()))
        bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)

        european_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, self.strike),
            ql.EuropeanExercise(today + int(self.expiry * 365))
        )
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

        ql_price = european_option.NPV()
        self.assertAlmostEqual(my_price, ql_price, places=2)

    def test_american_put_matches_quantlib(self):
        my_option = AmericanOption(self.expiry, self.strike, 'put', self.volatility, self.rate, steps=500)
        my_price = my_option.price(self.spot_price)

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_price))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, self.dividend_yield, ql.Actual365Fixed()))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.volatility, ql.Actual365Fixed()))
        bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)

        american_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, self.strike),
            ql.AmericanExercise(today, today + int(self.expiry * 365))
        )
        american_option.setPricingEngine(ql.BinomialVanillaEngine(bs_process, "crr", 500))

        ql_price = american_option.NPV()
        self.assertAlmostEqual(my_price, ql_price, places=1)

if __name__ == '__main__':
    unittest.main()