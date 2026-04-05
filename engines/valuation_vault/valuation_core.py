import numpy as np

class ActiveBusinessModel:
    """
    ABM V6.0: The Complete Dynamic Diagnostic Engine.
    Simulates institutional decision-making with Macro, Quali, and Quanti layers.
    """
    def __init__(self, ticker, sector, strategy_type='QG', macro_sentiment=0.0):
        self.ticker = ticker
        self.sector = sector
        self.strategy = strategy_type
        # Tier 2: D_Konjunktur clipped between -1.0 (Panic) and +1.0 (Euphoria)
        self.D_konjunktur = max(-1.0, min(macro_sentiment, 1.0)) 
        
        # Tier 1: The Triage (Base Weights)
        self.strategies = {
            'QG':  {'quali': 0.60, 'quanti': 0.40},
            'VAL': {'quali': 0.30, 'quanti': 0.70},
            'VC':  {'quali': 0.50, 'quanti': 0.50},
            'CYC': {'quali': 0.40, 'quanti': 0.60},
            'QI':  {'quali': 0.45, 'quanti': 0.55}
        }
        
        if self.strategy not in self.strategies:
            raise ValueError(f"Strategy {self.strategy} is not supported.")

    def _get_dynamic_weights(self):
        """Calculates macro-adjusted weights for Quality vs Quantity."""
        base_w = self.strategies[self.strategy]
        w_quali_adj = max(0.0, min(base_w['quali'] + (0.1 * self.D_konjunktur), 1.0))
        w_quanti_adj = 1.0 - w_quali_adj
        return w_quali_adj, w_quanti_adj

    def _calc_quali_score(self, moat, mgmt, market):
        """Tier 3: Qualitative Diagnose (Inputs 0-10)"""
        # Weights: 45% Moat, 30% Mgmt, 25% Market
        s_quali = (moat * 0.45) + (mgmt * 0.30) + (market * 0.25)
        return s_quali

    def _calc_quanti_score(self, fin):
        """Tier 4: The Quantitative Engine with Sector Toggles (Outputs 0-10)"""
        
        # Pillar 1: Growth (25%) -> Interpolate CAGR between 0% and 20%
        cagr = fin.get('cagr', 0.0)
        p1_growth = np.interp(cagr, [0.0, 0.20], [0, 10])

        # Pillar 2 & 3: Profitability & Balance Sheet (Sector Toggle applied)
        if self.sector == 'Financials':
            # Financials Logic (Banks/Insurances)
            roe = fin.get('roe', 0.0)
            cet1 = fin.get('cet1_ratio', 0.0)
            p2_profit = np.interp(roe, [0.05, 0.15], [0, 10])        # ROE 5% to 15%
            p3_balance = np.interp(cet1, [0.08, 0.13], [0, 10])      # CET1 8% to 13%
        else:
            # Standard Logic (Tech/Industrials)
            spread = fin.get('roic', 0.0) - fin.get('wacc', 0.0)
            debt_ebitda = fin.get('net_debt_ebitda', 5.0)
            p2_profit = np.interp(spread, [0.0, 0.10], [0, 10])      # Spread 0% to 10%
            p3_balance = np.interp(debt_ebitda, [0.0, 5.0], [10, 0]) # Note: Inverted! Low debt = High score

        # Pillar 4: Valuation (Fuzzy Logic)
        dcf_upside = fin.get('dcf_upside', 0.0)
        peg = fin.get('peg', 2.0)
        pe_ratio_to_hist = fin.get('pe_current', 15) / max(fin.get('pe_hist_5y', 15), 1)

        s_dcf = np.interp(dcf_upside, [-0.4, 0.0, 0.30], [0, 5, 10])
        s_peg = np.interp(peg, [0.8, 2.5], [10, 0])                  # Inverted: Low PEG = High score
        s_hist = np.interp(pe_ratio_to_hist, [0.7, 1.0, 1.3], [10, 5, 0]) 
        
        p4_valuation = (s_dcf + s_peg + s_hist) / 3

        # Combine the 4 pillars (25% each)
        s_quanti = (p1_growth + p2_profit + p3_balance + p4_valuation) / 4
        
        return s_quanti, {
            "Growth": p1_growth, "Profitability": p2_profit, 
            "BalanceSheet": p3_balance, "Valuation": p4_valuation
        }

    def _apply_risk_malus(self, gross_score, risks):
        """Tier 5: Subtractive Risk Penalties and Veto Logic"""
        if risks.get('extreme_insolvency', False):
            return gross_score * 0.10  # VETO!

        penalty = 0.0
        if risks.get('high_debt_reg', False): penalty += 1.00
        if risks.get('concentration', False): penalty += 0.50
        if risks.get('industry_risk', False): penalty += 0.25
            
        return max(0.0, gross_score - penalty)

    def run_master_engine(self, quali_inputs, financials, risks):
        """Executes the complete ABM V6.0 formula and returns the diagnostic profile."""
        
        w_quali, w_quanti = self._get_dynamic_weights()
        s_quali = self._calc_quali_score(**quali_inputs)
        s_quanti, quanti_breakdown = self._calc_quanti_score(financials)
        
        # The Master Equation
        s_brutto = (w_quali * s_quali) + (w_quanti * s_quanti)
        s_netto = self._apply_risk_malus(s_brutto, risks)
        
        return {
            "Identity": f"Strategy: {self.strategy} | Macro: {self.D_konjunktur}",
            "Weights": f"Quali: {w_quali*100:.0f}% | Quanti: {w_quanti*100:.0f}%",
            "S_Quali": round(s_quali, 2),
            "S_Quanti": round(s_quanti, 2),
            "S_Netto (Final Score)": round(s_netto, 2),
            "Quanti_DNA": {k: round(v, 2) for k, v in quanti_breakdown.items()}
        }

if __name__ == "__main__":
    # --- TEST RUN: Apple vs JPMorgan ---
    
    quali_data = {'moat': 9.0, 'mgmt': 8.5, 'market': 8.0}
    risks_data = {'industry_risk': True} # -0.25 penalty
    
    # 1. Standard Tech Stock (AAPL) - High ROIC, Low Debt
    aapl_fin = {
        'cagr': 0.15, 'roic': 0.25, 'wacc': 0.08, 'net_debt_ebitda': 0.5,
        'dcf_upside': 0.10, 'peg': 1.5, 'pe_current': 28, 'pe_hist_5y': 25
    }
    abm_aapl = ActiveBusinessModel("AAPL", sector="Technology", strategy_type="QG", macro_sentiment=0.5)
    
    # 2. Financial Stock (JPM) - Uses ROE and CET1 instead of ROIC/Debt
    jpm_fin = {
        'cagr': 0.08, 'roe': 0.14, 'cet1_ratio': 0.12, 
        'dcf_upside': 0.05, 'peg': 1.1, 'pe_current': 11, 'pe_hist_5y': 10
    }
    abm_jpm = ActiveBusinessModel("JPM", sector="Financials", strategy_type="QI", macro_sentiment=-0.5)

    print("\n🍏 APPLE (QG / Euphoria Regime):")
    for k, v in abm_aapl.run_master_engine(quali_data, aapl_fin, risks_data).items():
        print(f"  {k}: {v}")

    print("\n🏦 JPMORGAN (QI / Panic Regime):")
    for k, v in abm_jpm.run_master_engine(quali_data, jpm_fin, risks_data).items():
        print(f"  {k}: {v}")
        
    

class PositionSizer:
    """
    Translates an ABM diagnostic score into a concrete position size recommendation.

    Logic:
      The ABM score (0-10) represents the conviction level for a trade.
      We combine this with portfolio-level risk capacity (available CVaR budget)
      to derive a Kelly-inspired position size.

      Full Kelly is mathematically optimal but too aggressive for live trading.
      We use Half-Kelly (standard institutional practice) as a safety buffer.

    Formula:
      edge      = (abm_score / 10)           -> normalized conviction [0, 1]
      win_rate  = 0.5 + (edge * 0.25)        -> estimated hit rate [50%–75%]
      avg_win   = 1 + (edge * 0.5)           -> estimated reward ratio [1.0x–1.5x]
      avg_loss  = 1.0                         -> normalized loss unit
      kelly_pct = win_rate - (1 - win_rate) / (avg_win / avg_loss)
      half_kelly= kelly_pct * 0.5            -> conservative institutional standard
      
    The result is capped by:
      - Max single position: 25% of portfolio (concentration limit)
      - Min meaningful size: 2% (below this, the position doesn't move the needle)
      - CVaR budget check: if adding this position would breach the risk limit, scale down
    """

    MAX_POSITION = 0.25   # Hard cap: no single position > 25%
    MIN_POSITION = 0.02   # Floor: below 2% is noise
    RISK_FREE_RATE = 0.04

    def calculate(self, abm_score, portfolio_value, current_cvar_pct,
                  max_tolerable_cvar=-0.08, asset_volatility=None):
        """
        Returns a full position sizing recommendation dict.

        Parameters:
          abm_score           : float, the S_Netto from ABM (0-10)
          portfolio_value     : float, total current book value in $
          current_cvar_pct    : float, current portfolio CVaR (negative, e.g. -0.05)
          max_tolerable_cvar  : float, fund's hard risk limit (default -8%)
          asset_volatility    : float or None, annualized vol of proposed asset (optional)
        """
        # --- Step 1: Conviction edge from ABM score ---
        edge = abm_score / 10.0  # normalize to [0, 1]

        # --- Step 2: Estimate win rate and reward ratio ---
        # A score of 5/10 (neutral) implies ~50% win rate.
        # A score of 10/10 (exceptional) implies ~75% win rate.
        win_rate = 0.5 + (edge * 0.25)
        avg_win = 1.0 + (edge * 0.5)   # reward ratio vs 1.0 loss unit
        avg_loss = 1.0

        # --- Step 3: Full Kelly fraction ---
        loss_rate = 1.0 - win_rate
        kelly_full = win_rate - (loss_rate / (avg_win / avg_loss))
        kelly_full = max(0.0, kelly_full)  # Kelly can be negative (= don't trade)

        # --- Step 4: Half-Kelly (institutional standard) ---
        kelly_half = kelly_full * 0.5

        # --- Step 5: CVaR budget adjustment ---
        # How much CVaR headroom do we still have?
        cvar_headroom = abs(max_tolerable_cvar) - abs(current_cvar_pct)
        cvar_headroom = max(0.0, cvar_headroom)

        # If asset volatility is known, scale down if it would likely consume
        # more than 50% of the remaining CVaR headroom
        if asset_volatility and cvar_headroom > 0:
            # Rough estimate: a position of size w in an asset with daily vol v
            # contributes ~w * v * 1.645 to portfolio CVaR (parametric approximation)
            max_w_from_cvar = (cvar_headroom * 0.5) / (asset_volatility * 1.645)
            kelly_half = min(kelly_half, max_w_from_cvar)

        # --- Step 6: Apply hard caps ---
        final_weight = max(self.MIN_POSITION, min(self.MAX_POSITION, kelly_half))

        # --- Step 7: Translate to dollar amount ---
        position_dollar = final_weight * portfolio_value

        # --- Step 8: Generate qualitative recommendation ---
        if abm_score >= 8.0:
            action = "STRONG BUY"
            conviction = "High Conviction"
        elif abm_score >= 6.5:
            action = "BUY"
            conviction = "Moderate Conviction"
        elif abm_score >= 5.0:
            action = "HOLD / SMALL POSITION"
            conviction = "Low Conviction"
        elif abm_score >= 3.0:
            action = "AVOID"
            conviction = "Insufficient Edge"
        else:
            action = "DO NOT TRADE"
            conviction = "Negative Expected Value"

        return {
            "action":           action,
            "conviction":       conviction,
            "abm_score":        round(abm_score, 2),
            "kelly_full_pct":   round(kelly_full * 100, 1),
            "kelly_half_pct":   round(kelly_half * 100, 1),
            "recommended_weight": round(final_weight * 100, 1),
            "recommended_dollar": round(position_dollar, 2),
            "win_rate_est":     round(win_rate * 100, 1),
            "edge":             round(edge * 100, 1),
        }