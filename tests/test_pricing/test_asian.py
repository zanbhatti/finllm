import unittest
import math
import numpy as np

from finllm.pricing.black_scholes import EuropeanOption
from finllm.pricing.asian import AsianOption

try:
    import QuantLib as ql
    QL_AVAILABLE = True
except Exception:
    QL_AVAILABLE = False


def _ql_flat_curve(rate, day_count, ref_date):
    """Flat, continuously-compounded curve for r or q."""
    return ql.FlatForward(
        ref_date,
        ql.QuoteHandle(ql.SimpleQuote(rate)),
        day_count,
        ql.Continuous,
        ql.Annual,
    )


def _build_fixing_dates_calendar(ref_date, maturity_date, n):
    """
    n equally spaced fixing dates in CALENDAR days on (0, T], i.e., exclude S0, include maturity.
    This mirrors MC averaging over equally spaced times t_i = i/n * T.
    """
    total_days = maturity_date.serialNumber() - ref_date.serialNumber()  # calendar days
    if total_days <= 0:
        raise ValueError("Maturity must be after ref_date.")
    dates = []
    for i in range(1, n + 1):
        # round to nearest calendar day
        delta = int(round(i * total_days / n))
        d = ql.Date(ref_date.serialNumber() + delta)
        if d > maturity_date:
            d = maturity_date
        # ensure strictly increasing
        if dates and d <= dates[-1]:
            d = ql.Date(dates[-1].serialNumber() + 1)
            if d > maturity_date:
                d = maturity_date
        dates.append(d)
    dates[-1] = maturity_date
    return dates


@unittest.skipUnless(QL_AVAILABLE, "QuantLib not installed; skipping QuantLib parity tests.")
class TestAsianOptionAgainstQuantLib(unittest.TestCase):
    def setUp(self):
        # Market inputs (match your MC setup)
        self.S0    = 100.0
        self.K     = 100.0
        self.T     = 1.00          # years
        self.sigma = 0.20
        self.r     = 0.05
        self.q     = 0.00

        # MC / monitoring
        self.n_steps = 50
        self.n_sims  = 120_000
        self.seed    = 123456

        # Dates / conventions (fixed for determinism)
        self.calendar  = ql.TARGET()
        self.day_count = ql.Actual365Fixed()

        self.eval_date = ql.Date(1, ql.September, 2025)
        ql.Settings.instance().evaluationDate = self.eval_date

        # Use CALENDAR days to make maturity roughly T=1.0 year under Actual365Fixed
        total_calendar_days = int(round(self.T * 365))
        self.maturity_date = ql.Date(self.eval_date.serialNumber() + total_calendar_days)

        # Curves (flat, continuous)
        self.r_ts = ql.YieldTermStructureHandle(_ql_flat_curve(self.r, self.day_count, self.eval_date))
        self.q_ts = ql.YieldTermStructureHandle(_ql_flat_curve(self.q, self.day_count, self.eval_date))
        self.vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.eval_date, self.calendar, self.sigma, self.day_count)
        )

        # Underlying process: (spot, dividendYield=q, riskFreeRate=r, blackVol)
        self.spot_h = ql.QuoteHandle(ql.SimpleQuote(self.S0))
        self.process = ql.BlackScholesMertonProcess(self.spot_h, self.q_ts, self.r_ts, self.vol_ts)

        # Fixing dates (exclude S0, include maturity), equally spaced in CALENDAR days
        self.fixing_dates = _build_fixing_dates_calendar(self.eval_date, self.maturity_date, self.n_steps)

    def _ql_discrete_arith_price_asian_call(self):
        """
        QuantLib MC price for discrete arithmetic-average price Asian call with same grid.
        """
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.K)
        exercise = ql.EuropeanExercise(self.maturity_date)

        running_acc = 0.0
        past_fixings = 0
        avg_type = ql.Average.Arithmetic
        option = ql.DiscreteAveragingAsianOption(
            avg_type, running_acc, past_fixings, self.fixing_dates, payoff, exercise
        )

        engine = ql.MCDiscreteArithmeticAPEngine(
            self.process,
            "pseudorandom",
            antitheticVariate=True,
            requiredSamples=self.n_sims,
            seed=int(self.seed),
        )
        option.setPricingEngine(engine)
        return option.NPV()

    def test_asian_call_mc_matches_quantlib_within_mc_error(self):
        # Our MC price
        asian = AsianOption(
            self.T, self.K, 'call', self.sigma, self.r,
            dividend_yield=self.q,
            num_simulations=self.n_sims,
            num_steps=self.n_steps,
            antithetic=True,
            seed=self.seed
        )
        our_price, our_se = asian.price(self.S0)

        # QuantLib MC price
        ql_price = self._ql_discrete_arith_price_asian_call()

        # Allow difference within a multiple of our std error (and a small floor)
        tol = max(0.006, 3.5 * our_se)
        self.assertAlmostEqual(
            our_price, ql_price, delta=tol,
            msg=f"Our MC: {our_price:.6f} ± {our_se:.6f} vs QL: {ql_price:.6f} (tol={tol:.6f})"
        )

    def test_limit_num_steps_equals_european(self):
        """
        With exactly one monitoring date at T (exclude S0), the Asian arithmetic average equals S_T,
        so the Asian call equals the European call.
        We verify equality both in our pricer and in QuantLib.
        """
        # Our Asian (1 step) vs our European
        asian_1 = AsianOption(
            self.T, self.K, 'call', self.sigma, self.r, dividend_yield=self.q,
            num_simulations=self.n_sims, num_steps=1, antithetic=True, seed=self.seed
        )
        a_price, a_se = asian_1.price(self.S0)

        euro = EuropeanOption(self.T, self.K, 'call', self.sigma, self.r, self.q)
        e_price = euro.price(self.S0)  # scalar expected

        tol = max(0.005, 3.0 * a_se)
        self.assertAlmostEqual(a_price, e_price, delta=tol)

        # QuantLib’s European analytic for cross-check
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.K)
        exercise = ql.EuropeanExercise(self.maturity_date)
        european = ql.VanillaOption(payoff, exercise)
        european.setPricingEngine(ql.AnalyticEuropeanEngine(self.process))
        ql_euro = european.NPV()

        # Your BS class should match analytic BS tightly
        self.assertAlmostEqual(e_price, ql_euro, delta=1e-10)
