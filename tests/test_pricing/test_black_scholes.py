import unittest
import math
from finllm.pricing.black_scholes import EuropeanOption

try:
    import QuantLib as ql
    QL_AVAILABLE = True
except Exception:
    QL_AVAILABLE = False


def bs_call_closed_form(S0, K, r, q, sigma, T):
    """Ground-truth Black–Scholes–Merton call."""
    from math import log, sqrt, exp
    from statistics import NormalDist
    N = NormalDist().cdf
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)


def bs_put_closed_form(S0, K, r, q, sigma, T):
    """Ground-truth Black–Scholes–Merton put."""
    from math import log, sqrt, exp
    from statistics import NormalDist
    N = NormalDist().cdf
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * N(-d2) - S0 * exp(-q * T) * N(-d1)


if QL_AVAILABLE:
    def make_bsm_process(S0, r, q, sigma, eval_date,
                         calendar=ql.TARGET(), day_count=ql.Actual365Fixed()):
        """Flat, continuously compounded r and q; constant vol."""
        ql.Settings.instance().evaluationDate = eval_date

        spot = ql.QuoteHandle(ql.SimpleQuote(S0))
        r_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, ql.QuoteHandle(ql.SimpleQuote(r)),
                           day_count, ql.Continuous, ql.Annual)
        )
        q_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, ql.QuoteHandle(ql.SimpleQuote(q)),
                           day_count, ql.Continuous, ql.Annual)
        )
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(eval_date, calendar, sigma, day_count)
        )
        # ORDER: (spot, dividendYield, riskFreeRate, blackVol)
        return ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)


class TestEuropeanOption(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.sigma = 0.2
        self.r = 0.05
        self.q = 0.0

    def test_call_price_close_to_expected(self):
        opt = EuropeanOption(self.T, self.K, 'call', self.sigma, self.r, self.q)
        price = opt.price(self.S0)
        self.assertAlmostEqual(price, 10.4505835722, places=6)

    def test_put_price_close_to_expected(self):
        opt = EuropeanOption(self.T, self.K, 'put', self.sigma, self.r, self.q)
        price = opt.price(self.S0)
        self.assertAlmostEqual(price, 5.5735260223, places=6)

    def test_put_call_parity(self):
        call = EuropeanOption(self.T, self.K, 'call', self.sigma, self.r, self.q)
        put  = EuropeanOption(self.T, self.K, 'put',  self.sigma, self.r, self.q)
        lhs = call.price(self.S0) - put.price(self.S0)
        rhs = self.S0 * math.exp(-self.q * self.T) - self.K * math.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_invalid_option_type(self):
        with self.assertRaises((ValueError, AssertionError)):
            EuropeanOption(self.T, self.K, 'banana', self.sigma, self.r, self.q)

    @unittest.skipUnless(QL_AVAILABLE, "QuantLib not installed")
    def test_call_matches_quantlib(self):
        my_price = EuropeanOption(self.T, self.K, 'call', self.sigma, self.r, self.q).price(self.S0)

        today = ql.Date(1, ql.September, 2025)
        process = make_bsm_process(self.S0, self.r, self.q, self.sigma, today)
        maturity = today + int(self.T * 365)

        ql_opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, self.K),
            ql.EuropeanExercise(maturity)
        )
        ql_opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        ql_price = ql_opt.NPV()

        self.assertAlmostEqual(my_price, ql_price, places=6)

    @unittest.skipUnless(QL_AVAILABLE, "QuantLib not installed")
    def test_put_matches_quantlib(self):
        my_price = EuropeanOption(self.T, self.K, 'put', self.sigma, self.r, self.q).price(self.S0)

        today = ql.Date(1, ql.September, 2025)
        process = make_bsm_process(self.S0, self.r, self.q, self.sigma, today)
        maturity = today + int(self.T * 365)

        ql_opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, self.K),
            ql.EuropeanExercise(maturity)
        )
        ql_opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        ql_price = ql_opt.NPV()

        self.assertAlmostEqual(my_price, ql_price, places=6)


if __name__ == "__main__":
    unittest.main()
