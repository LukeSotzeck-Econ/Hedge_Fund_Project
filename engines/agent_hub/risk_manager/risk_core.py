import numpy as np
import pandas as pd

class RiskGatekeeper:
    """
    Deterministic Risk Engine utilizing Historical Simulation.
    Evaluates absolute CVaR and marginal risk contribution of proposed allocations.
    """
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.tail_threshold = 1.0 - confidence_level

    def calculate_portfolio_risk(self, prices_df, weights, portfolio_value):
        """Computes VaR and CVaR using vector dot products of historical returns."""
        asset_returns = prices_df.pct_change().dropna()
        portfolio_returns = asset_returns.dot(weights)
        
        var_pct = float(portfolio_returns.quantile(self.tail_threshold))
        tail_losses = portfolio_returns[portfolio_returns <= var_pct]
        cvar_pct = float(tail_losses.mean())
        
        var_value = portfolio_value * abs(var_pct)
        cvar_value = portfolio_value * abs(cvar_pct)
        
        return var_pct, var_value, cvar_pct, cvar_value

    def evaluate_trade_impact(self, current_holdings, proposed_ticker, proposed_amount, prices_df):
        """
        Calculates the Marginal CVaR (Before & After) of adding a new position.
        current_holdings: dict mapping ticker to current monetary exposure.
        proposed_ticker: str of the asset being evaluated.
        proposed_amount: float representing the monetary value of the proposed trade.
        """
        # 1. State A: Current Portfolio
        current_value = sum(current_holdings.values())
        current_tickers = list(current_holdings.keys())
        current_weights = np.array([current_holdings[t] / current_value for t in current_tickers])
        
        # Isolate the price matrix for existing assets
        current_prices = prices_df[current_tickers]
        _, _, current_cvar_pct, current_cvar_val = self.calculate_portfolio_risk(
            current_prices, current_weights, current_value
        )
        
        # 2. State B: Proposed Portfolio
        new_holdings = current_holdings.copy()
        if proposed_ticker in new_holdings:
            new_holdings[proposed_ticker] += proposed_amount
        else:
            new_holdings[proposed_ticker] = proposed_amount
            
        new_value = sum(new_holdings.values())
        new_tickers = list(new_holdings.keys())
        new_weights = np.array([new_holdings[t] / new_value for t in new_tickers])
        
        # Isolate the price matrix for the new asset mix
        new_prices = prices_df[new_tickers]
        _, _, new_cvar_pct, new_cvar_val = self.calculate_portfolio_risk(
            new_prices, new_weights, new_value
        )
        
        # 3. Risk Attribution (Marginal Delta)
        # Note: CVaR is negative in this context, so a positive delta means risk increased (more negative)
        marginal_cvar_pct = new_cvar_pct - current_cvar_pct
        marginal_cvar_val = new_cvar_val - current_cvar_val
        
        return {
            "current_cvar_pct": current_cvar_pct,
            "current_cvar_val": current_cvar_val,
            "new_cvar_pct": new_cvar_pct,
            "new_cvar_val": new_cvar_val,
            "marginal_cvar_pct": marginal_cvar_pct,
            "marginal_cvar_val": marginal_cvar_val,
            "new_portfolio_value": new_value
        }

    def veto_check(self, new_cvar_pct, max_tolerable_loss=-0.08):
        """Circuit breaker evaluating the post-trade portfolio state."""
        if new_cvar_pct < max_tolerable_loss:
            return True, f"VETO: Proposed CVaR ({new_cvar_pct*100:.2f}%) breaches fund limit ({max_tolerable_loss*100:.2f}%)"
        return False, "TRADE APPROVED: Marginal risk within acceptable bounds."


if __name__ == "__main__":
    import yfinance as yf  # Only needed for this standalone test
    # --- ISOLATED COMPONENT TEST ---
    print("Initializing Marginal Risk Evaluation...\n")
    
    # Base portfolio state
    current_book = {"JPM": 90000.0}
    
    # Proposed trade
    target_asset = "TSLA"
    trade_size = 10000.0
    
    # We must fetch data for all assets in the combined universe
    all_tickers = list(current_book.keys()) + [target_asset]
    # Ensure unique tickers in case we are adding to an existing position
    all_tickers = list(set(all_tickers)) 
    
    print(f"Fetching 5Y historical matrix for universe: {all_tickers}...")
    data = yf.download(all_tickers, period="5y", progress=False)['Close']
    
    # Execute Gatekeeper
    gatekeeper = RiskGatekeeper(confidence_level=0.95)
    impact = gatekeeper.evaluate_trade_impact(current_book, target_asset, trade_size, data)
    
    print("\n--- RISK ATTRIBUTION REPORT ---")
    print(f"Current Book Value : ${sum(current_book.values()):,.2f}")
    print(f"Proposed Trade     : +${trade_size:,.2f} {target_asset}")
    print(f"New Book Value     : ${impact['new_portfolio_value']:,.2f}\n")
    
    print(f"Current Portfolio CVaR : {impact['current_cvar_pct']*100:.2f}% | ${impact['current_cvar_val']:,.2f}")
    print(f"Proposed Portfolio CVaR: {impact['new_cvar_pct']*100:.2f}% | ${impact['new_cvar_val']:,.2f}")
    
    # A negative marginal percentage indicates the portfolio became riskier
    print(f"\nMarginal Risk Delta    : {impact['marginal_cvar_pct']*100:.2f}%")
    
    # Enforce Leuchtturm constraints
    is_vetoed, msg = gatekeeper.veto_check(impact['new_cvar_pct'], max_tolerable_loss=-0.08)
    print(f"\nGATEKEEPER STATUS: {msg}")