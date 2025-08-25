import unittest
from finllm.pricing.black_scholes import EuropeanOption
from finllm.pricing.american import AmericanOption
from finllm.pricing.bermudan import BermudanOption
import QuantLib as ql


class TestBermudanOption(unittest.TestCase):
    def setUp(self):
        self.spot = 100
        self.strike = 100
        self.expiry = 1.0
        self.vol = 0.2
        self.rate = 0.05
        self.div_yield = 0.0

    def test_bermudan_price_between_european_and_american(self):
        # Put: Bermudan should lie between European and American
        euro_option = EuropeanOption(self.expiry, self.strike, 'put', self.vol, self.rate)
        amer_option = AmericanOption(self.expiry, self.strike, 'put', self.vol, self.rate, steps=200)
        # One early exercise date at T/2, no exercise at maturity unless included
        berm_option = BermudanOption(
            self.expiry, self.strike, 'put', self.vol, self.rate,
            steps=200, exercise_dates=[0.5, self.expiry]  # include maturity exercise
        )

        euro_price = euro_option.price(self.spot)
        amer_price = amer_option.price(self.spot)
        berm_price = berm_option.price(self.spot)

        self.assertGreaterEqual(berm_price, euro_price)
        self.assertLessEqual(berm_price, amer_price)

    def test_bermudan_with_no_early_exercise_equals_european(self):
        # Call, no dividends: if maturity is the ONLY exercise date, Bermudan ≈ European
        euro_option = EuropeanOption(self.expiry, self.strike, 'call', self.vol, self.rate)
        berm_option = BermudanOption(
            self.expiry, self.strike, 'call', self.vol, self.rate,
            steps=1000, exercise_dates=[self.expiry]  # explicitly allow maturity exercise
        )

        euro_price = euro_option.price(self.spot)
        berm_price = berm_option.price(self.spot)

        # Lattice vs closed-form: allow slight discretization
        self.assertAlmostEqual(berm_price, euro_price, places=2)

    # --- Helper that aligns QuantLib’s exercise dates with our lattice snapping ---
    def bermudan_test_helper(self, option_type, steps, places, exercise_dates):
        """
        exercise_dates are given as fractions of expiry (e.g., [0.25, 0.5, 0.75]).
        We map them to the same lattice steps as our implementation (round to nearest step),
        and we DO NOT include expiry unless explicitly provided by the caller.
        """
        # Our model
        my_option = BermudanOption(
            self.expiry, self.strike, option_type, self.vol, self.rate,
            dividend_yield=self.div_yield, steps=steps, exercise_dates=exercise_dates
        )
        my_price = my_option.price(self.spot)

        # QuantLib setup (flat curves, Black–Scholes–Merton process)
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        spot = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        rate = ql.YieldTermStructureHandle(ql.FlatForward(today, self.rate, ql.Actual365Fixed()))
        div  = ql.YieldTermStructureHandle(ql.FlatForward(today, self.div_yield, ql.Actual365Fixed()))
        vol  = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), self.vol, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(spot, div, rate, vol)

        # Map fractions -> times -> SNAP to lattice steps using the SAME rule as our code.
        # Our implementation uses round() when deciding step membership; replicate that here.
        T = self.expiry
        dt = T / steps
        # Treat inputs as fractions of T
        raw_times = [t * T for t in exercise_dates]

        # Snap to nearest step index with round; exclude maturity unless explicitly provided
        snapped_times = []
        for t in raw_times:
            k = int(round(t / dt))
            if k < 0:
                k = 0
            if k > steps:
                k = steps
            # Only include maturity if the user explicitly asked for expiry in exercise_dates
            if k == steps and (abs(t - T) > 1e-12):
                # This time rounded to maturity but caller didn't explicitly include expiry
                continue
            # Exclude step 0 and duplicates
            if 1 <= k <= steps - 1:
                snapped_times.append(k * dt)
            elif k == steps and any(abs(t - T) <= 1e-12 for t in raw_times):
                # include maturity only if explicitly present in input list
                snapped_times.append(T)

        # Convert snapped times to QuantLib Dates
        ql_exercise_dates = [today + int(round(t * 365)) for t in snapped_times]

        ql_option = ql.VanillaOption(
            ql.PlainVanillaPayoff(
                ql.Option.Put if option_type == 'put' else ql.Option.Call,
                self.strike
            ),
            ql.BermudanExercise(ql_exercise_dates)
        )
        ql_option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))

        self.assertAlmostEqual(my_price, ql_option.NPV(), places=places)

    def test_put_matches_quantlib(self):
        # NOTE: no expiry in the list; exercise ONLY at the three internal dates
        self.bermudan_test_helper('put', 500, 2, [0.25, 0.5, 0.75])

    def test_call_matches_quantlib(self):
        # NOTE: no expiry in the list; exercise ONLY at the three internal dates
        self.bermudan_test_helper('call', 500, 2, [0.25, 0.5, 0.75])
