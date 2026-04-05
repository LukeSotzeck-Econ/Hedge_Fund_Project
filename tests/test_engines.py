import pytest
import numpy as np
import pandas as pd
import sys
import os

# conftest.py in the project root handles sys.path — this line is a fallback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engines.valuation_vault.valuation_core import ActiveBusinessModel
from engines.agent_hub.risk_manager.risk_core import RiskGatekeeper


# ─────────────────────────────────────────────
#  FIXTURES  (shared test data, created once)
# ─────────────────────────────────────────────

@pytest.fixture
def standard_quali():
    """Solid qualitative inputs — used across multiple tests."""
    return {'moat': 8.0, 'mgmt': 7.0, 'market': 7.5}

@pytest.fixture
def no_risks():
    """Clean risk profile — no penalties applied."""
    return {'extreme_insolvency': False, 'high_debt_reg': False,
            'concentration': False, 'industry_risk': False}

@pytest.fixture
def fake_price_data():
    """
    Synthetic price matrix for 3 assets over 300 days.
    Deterministic random seed so tests always produce the same values.
    """
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=300, freq='B')
    prices = pd.DataFrame(
        {
            'JPM':  100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 300)),
            'AAPL': 150 * np.cumprod(1 + np.random.normal(0.0006, 0.018, 300)),
            'TSLA': 200 * np.cumprod(1 + np.random.normal(0.0003, 0.030, 300)),
        },
        index=dates
    )
    return prices


# ─────────────────────────────────────────────
#  ABM: VALUATION ENGINE TESTS
# ─────────────────────────────────────────────

class TestActiveBusinessModel:

    def test_tech_score_is_high_for_strong_fundamentals(self, standard_quali, no_risks):
        """
        A tech stock with high ROIC, low debt, and a cheap valuation
        should score above 7.0 — that's the whole point of the engine.
        """
        strong_tech_fin = {
            'cagr': 0.18, 'roic': 0.28, 'wacc': 0.08, 'net_debt_ebitda': 0.3,
            'dcf_upside': 0.25, 'peg': 1.2, 'pe_current': 22, 'pe_hist_5y': 25
        }
        abm = ActiveBusinessModel("AAPL", sector="Technology", strategy_type="QG", macro_sentiment=0.0)
        report = abm.run_master_engine(standard_quali, strong_tech_fin, no_risks)

        assert report['S_Netto (Final Score)'] > 7.0, (
            f"Expected strong tech stock to score > 7.0, got {report['S_Netto (Final Score)']}"
        )

    def test_financials_sector_uses_roe_not_roic(self, standard_quali, no_risks):
        """
        Critical sector-toggle test.
        A Financials stock must be scored on ROE/CET1 — NOT on ROIC/Debt.
        We pass ONLY roe and cet1_ratio (no roic key), which would return 0.0
        if the toggle is broken. A passing score > 4.0 proves the branch works.
        """
        bank_fin = {
            'cagr': 0.08, 'roe': 0.13, 'cet1_ratio': 0.115,
            'dcf_upside': 0.05, 'peg': 1.1, 'pe_current': 11, 'pe_hist_5y': 10
        }
        abm = ActiveBusinessModel("JPM", sector="Financials", strategy_type="QI", macro_sentiment=0.0)
        report = abm.run_master_engine(standard_quali, bank_fin, no_risks)

        assert report['S_Netto (Final Score)'] > 4.0, (
            "Financials sector toggle may be broken — score too low for solid ROE/CET1 inputs."
        )

    def test_insolvency_veto_collapses_score(self, standard_quali):
        """
        extreme_insolvency=True must trigger the VETO (score × 0.10).
        Even a perfect gross score of 10.0 becomes 1.0 — that's the circuit breaker.
        """
        perfect_fin = {
            'cagr': 0.20, 'roic': 0.30, 'wacc': 0.05, 'net_debt_ebitda': 0.0,
            'dcf_upside': 0.30, 'peg': 0.8, 'pe_current': 10, 'pe_hist_5y': 20
        }
        veto_risks = {'extreme_insolvency': True, 'high_debt_reg': False,
                      'concentration': False, 'industry_risk': False}

        abm = ActiveBusinessModel("DANGER", sector="Technology", strategy_type="QG", macro_sentiment=0.0)
        report = abm.run_master_engine(standard_quali, perfect_fin, veto_risks)

        assert report['S_Netto (Final Score)'] < 1.5, (
            f"Insolvency veto should collapse score below 1.5, got {report['S_Netto (Final Score)']}"
        )

    def test_macro_panic_shifts_weights_toward_quanti(self, standard_quali, no_risks):
        """
        In a Panic regime (macro_sentiment=-1.0), the QG strategy quali weight
        decreases. Base QG is 60% Quali — panic shifts it to 50%.
        With average quant inputs, the panic score must be lower.
        """
        average_fin = {
            'cagr': 0.08, 'roic': 0.12, 'wacc': 0.09, 'net_debt_ebitda': 2.0,
            'dcf_upside': 0.0, 'peg': 2.0, 'pe_current': 18, 'pe_hist_5y': 18
        }
        abm_normal = ActiveBusinessModel("X", sector="Technology", strategy_type="QG", macro_sentiment=0.0)
        abm_panic  = ActiveBusinessModel("X", sector="Technology", strategy_type="QG", macro_sentiment=-1.0)

        score_normal = abm_normal.run_master_engine(standard_quali, average_fin, no_risks)['S_Netto (Final Score)']
        score_panic  = abm_panic.run_master_engine(standard_quali, average_fin, no_risks)['S_Netto (Final Score)']

        assert score_panic < score_normal, (
            "Panic regime should reduce the score for a quali-heavy strategy when quant inputs are average."
        )

    def test_score_is_bounded_between_0_and_10(self, standard_quali, no_risks):
        """Scores must never go outside [0, 10] regardless of inputs."""
        extreme_fin = {
            'cagr': 99.0, 'roic': 99.0, 'wacc': 0.0, 'net_debt_ebitda': -99.0,
            'dcf_upside': 99.0, 'peg': 0.0, 'pe_current': 1, 'pe_hist_5y': 100
        }
        abm = ActiveBusinessModel("X", sector="Technology", strategy_type="QG", macro_sentiment=0.0)
        report = abm.run_master_engine(standard_quali, extreme_fin, no_risks)
        score = report['S_Netto (Final Score)']

        assert 0.0 <= score <= 10.0, f"Score {score} is out of bounds [0, 10]"


# ─────────────────────────────────────────────
#  RISK ENGINE TESTS
# ─────────────────────────────────────────────

class TestRiskGatekeeper:

    def test_cvar_is_negative(self, fake_price_data):
        """
        CVaR represents a loss — it must always be a negative number.
        A positive CVaR would mean we expect gains in the worst tail,
        which is mathematically nonsensical for this implementation.
        """
        gatekeeper = RiskGatekeeper(confidence_level=0.95)
        current_book = {'JPM': 60000.0, 'AAPL': 40000.0}
        impact = gatekeeper.evaluate_trade_impact(current_book, 'TSLA', 10000.0, fake_price_data)

        assert impact['new_cvar_pct'] < 0.0, "CVaR should be negative (represents a loss)"
        assert impact['current_cvar_pct'] < 0.0, "Current CVaR should be negative"

    def test_high_vol_asset_has_worse_standalone_cvar(self):
        """
        A portfolio of a single high-volatility asset must have a worse
        (more negative) CVaR than a portfolio of a single low-volatility asset.

        Fully deterministic hand-crafted price series — no random seed,
        no correlation surprises. Perfectly alternating returns:
          - low_vol:  ±0.5% daily  → mild tail losses
          - high_vol: ±3.0% daily  → severe tail losses
        """
        dates = pd.date_range('2022-01-01', periods=200, freq='B')

        low_vol_returns  = np.tile([0.005, -0.005], 100)
        high_vol_returns = np.tile([0.030, -0.030], 100)

        prices = pd.DataFrame({
            'LOW':  100 * np.cumprod(1 + low_vol_returns),
            'HIGH': 100 * np.cumprod(1 + high_vol_returns),
        }, index=dates)

        gatekeeper = RiskGatekeeper(confidence_level=0.95)

        impact_low  = gatekeeper.evaluate_trade_impact(
            {'LOW':  100000.0}, 'LOW',  0.01, prices
        )
        impact_high = gatekeeper.evaluate_trade_impact(
            {'HIGH': 100000.0}, 'HIGH', 0.01, prices
        )

        assert impact_high['new_cvar_pct'] < impact_low['new_cvar_pct'], (
            f"High-vol CVaR ({impact_high['new_cvar_pct']:.4f}) should be more "
            f"negative than low-vol CVaR ({impact_low['new_cvar_pct']:.4f})"
        )

    def test_veto_triggers_above_threshold(self, fake_price_data):
        """
        If CVaR breaches -8%, the gatekeeper must return is_vetoed=True.
        """
        gatekeeper = RiskGatekeeper(confidence_level=0.95)
        is_vetoed, msg = gatekeeper.veto_check(-0.12, max_tolerable_loss=-0.08)

        assert is_vetoed is True
        assert "VETO" in msg

    def test_veto_passes_within_threshold(self):
        """A CVaR of -3% should pass the -8% threshold cleanly."""
        gatekeeper = RiskGatekeeper(confidence_level=0.95)
        is_vetoed, msg = gatekeeper.veto_check(-0.03, max_tolerable_loss=-0.08)

        assert is_vetoed is False
        assert "APPROVED" in msg

    def test_portfolio_value_is_sum_of_holdings(self, fake_price_data):
        """
        The returned new_portfolio_value must equal the sum of all positions.
        This is an accounting identity — if it breaks, every dollar metric is wrong.
        """
        gatekeeper = RiskGatekeeper(confidence_level=0.95)
        current_book = {'JPM': 60000.0, 'AAPL': 40000.0}
        proposed_amount = 10000.0
        impact = gatekeeper.evaluate_trade_impact(
            current_book, 'TSLA', proposed_amount, fake_price_data
        )

        expected_value = 60000.0 + 40000.0 + proposed_amount
        assert impact['new_portfolio_value'] == pytest.approx(expected_value), (
            f"Portfolio value mismatch: expected {expected_value}, got {impact['new_portfolio_value']}"
        )