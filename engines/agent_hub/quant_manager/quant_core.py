import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class QuantModeler:
    """
    The Deterministic Modeler (Werkstatt).
    Calculates returns, volatility, solves for optimal allocation, and measures drawdowns.
    """
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252

    def calculate_returns(self, prices):
        return prices.pct_change().dropna()

    def calculate_drawdown(self, returns):
        """Calculates the Maximum Drawdown from a returns series."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())

    def evaluate_asset(self, asset_prices, benchmark_prices):
        """Standard quant profile for a single ticker."""
        asset_returns = self.calculate_returns(asset_prices)
        benchmark_returns = self.calculate_returns(benchmark_prices)
        
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        aligned_data.columns = ['Asset', 'Benchmark']
        
        total_compound_return = (1 + aligned_data['Asset']).prod()
        years = len(aligned_data) / self.trading_days
        annualized_return = (total_compound_return ** (1 / years)) - 1
        
        annualized_volatility = aligned_data['Asset'].std() * np.sqrt(self.trading_days)
        
        covariance_matrix = np.cov(aligned_data['Asset'], aligned_data['Benchmark'])
        beta = covariance_matrix[0, 1] / np.var(aligned_data['Benchmark'])
        
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        mdd = self.calculate_drawdown(aligned_data['Asset'])
        
        return {
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "beta": float(beta),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": mdd
        }

    def calculate_portfolio_metrics(self, prices_df, weights, benchmark_prices):
        """Aggregate performance for a weighted basket of assets."""
        asset_returns = prices_df.pct_change().dropna()
        benchmark_returns = self.calculate_returns(benchmark_prices)
        
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        clean_asset_returns = aligned_data.iloc[:, :-1]
        clean_bench_returns = aligned_data.iloc[:, -1]
        
        port_returns = clean_asset_returns.dot(weights)
        
        total_compound = (1 + port_returns).prod()
        years = len(port_returns) / self.trading_days
        ann_return = (total_compound ** (1 / years)) - 1
        ann_volatility = port_returns.std() * np.sqrt(self.trading_days)
        
        covar = np.cov(port_returns, clean_bench_returns)[0, 1]
        beta = covar / np.var(clean_bench_returns)
        sharpe = (ann_return - self.risk_free_rate) / ann_volatility
        mdd = self.calculate_drawdown(port_returns)
        
        return float(ann_return), float(ann_volatility), float(beta), float(sharpe), float(mdd)

    def evaluate_trade_impact(self, current_holdings, proposed_ticker, proposed_amount, prices_df, benchmark_prices):
        """Pre vs Post trade comparative analysis."""
        current_value = sum(current_holdings.values())
        current_tickers = list(current_holdings.keys())
        current_weights = np.array([current_holdings[t] / current_value for t in current_tickers])
        
        cur_ret, cur_vol, cur_beta, cur_sharpe, cur_mdd = self.calculate_portfolio_metrics(
            prices_df[current_tickers], current_weights, benchmark_prices
        )
        
        new_holdings = current_holdings.copy()
        new_holdings[proposed_ticker] = new_holdings.get(proposed_ticker, 0.0) + proposed_amount
        
        new_value = sum(new_holdings.values())
        new_tickers = list(new_holdings.keys())
        new_weights = np.array([new_holdings[t] / new_value for t in new_tickers])
        
        new_ret, new_vol, new_beta, new_sharpe, new_mdd = self.calculate_portfolio_metrics(
            prices_df[new_tickers], new_weights, benchmark_prices
        )
        
        return {
            "current_return": cur_ret, "current_vol": cur_vol, "current_sharpe": cur_sharpe, "current_mdd": cur_mdd,
            "new_return": new_ret, "new_vol": new_vol, "new_sharpe": new_sharpe, "new_mdd": new_mdd,
            "delta_return": new_ret - cur_ret,
            "delta_vol": new_vol - cur_vol,
            "delta_sharpe": new_sharpe - cur_sharpe,
            "delta_mdd": new_mdd - cur_mdd
        }

    def optimize_portfolio(self, prices_df):
        """Solver: Maximizes Sharpe Ratio by adjusting weights."""
        asset_returns = self.calculate_returns(prices_df)
        num_assets = len(prices_df.columns)
        
        def objective(weights):
            port_returns = asset_returns.dot(weights)
            ann_ret = np.sum(np.log1p(port_returns)) * (self.trading_days / len(port_returns))
            ann_ret = np.expm1(ann_ret)
            ann_vol = port_returns.std() * np.sqrt(self.trading_days)
            if ann_vol == 0: return 0
            sharpe = (ann_ret - self.risk_free_rate) / ann_vol
            return -sharpe

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(prices_df.columns, result.x)) if result.success else {t: 1.0/num_assets for t in prices_df.columns}

    def get_backtest_series(self, prices_df, weights):
        """Generates a cumulative return series for the portfolio."""
        returns = prices_df.pct_change().dropna()
        portfolio_returns = returns.dot(weights)
        cumulative_growth = (1 + portfolio_returns).cumprod()
        # Anchor start point on returns.index[0] (not prices index[0])
        # to keep the date axis consistent and avoid chart corruption.
        start = pd.Series([1.0], index=[returns.index[0]])
        return pd.concat([start, cumulative_growth.iloc[1:]])

    def get_correlation_matrix(self, prices_df):
        """Pearson correlation of daily returns between all assets."""
        return prices_df.pct_change().dropna().corr()

    def run_stress_test(self, prices_df, weights):
        """
        TWO-TRACK STRESS TEST ENGINE
        ─────────────────────────────
        Track A — Historical (data-dependent):
          Replays portfolio through actual crisis windows IF the price
          history covers that period. With 5y data, only Covid + Rate Shock
          will have real data. GFC and Dot-com will show as unavailable.

        Track B — Parametric Shocks (always available):
          Applies standardized instantaneous shocks to the portfolio's
          return distribution. These are calibrated to actual crisis
          magnitudes but don't require historical data from those periods.
          This is how banks run stress tests on new instruments with
          short track records — Basel III compliant approach.

          Scenarios:
            Mild Correction   : -10% market shock, vol ×1.5
            Bear Market       : -25% market shock, vol ×2.0
            Severe Crash      : -40% market shock, vol ×3.0  (GFC-calibrated)
            Tail Event        : -55% market shock, vol ×4.0  (Dot-com calibrated)
        """
        results = {}
        all_returns = prices_df.pct_change().dropna()

        # ── TRACK A: Historical windows ──────────────────────────────────
        historical_scenarios = {
            "Covid Crash (Feb–Mar 2020)":       ("2020-02-19", "2020-03-23"),
            "Rate Shock (Full Year 2022)":       ("2022-01-01", "2022-12-31"),
            "GFC (Sep 2008–Mar 2009)":           ("2008-09-01", "2009-03-31"),
            "Dot-com Bust (Mar 2000–Oct 2002)":  ("2000-03-01", "2002-10-31"),
        }

        historical_results = {}
        for name, (start, end) in historical_scenarios.items():
            try:
                window = all_returns.loc[start:end]
                if len(window) < 5:
                    historical_results[name] = {
                        "max_drawdown": None,
                        "total_return": None,
                        "trading_days": 0,
                        "note": "Outside price history window"
                    }
                    continue
                port_returns = window.dot(weights)
                cumulative = (1 + port_returns).cumprod()
                peak = cumulative.cummax()
                max_dd = float(((cumulative - peak) / peak).min())
                total_return = float(cumulative.iloc[-1] - 1.0)
                historical_results[name] = {
                    "max_drawdown": round(max_dd * 100, 2),
                    "total_return": round(total_return * 100, 2),
                    "trading_days": len(window),
                    "note": "Historical"
                }
            except Exception:
                historical_results[name] = {
                    "max_drawdown": None, "total_return": None,
                    "trading_days": 0, "note": "Insufficient data"
                }

        results["historical"] = historical_results

        # ── TRACK B: Parametric shocks (always available) ─────────────────
        # Build portfolio return distribution from available history
        port_daily_returns = all_returns.dot(weights)
        daily_vol = float(port_daily_returns.std())
        daily_mean = float(port_daily_returns.mean())

        # Each shock: (label, market_drop_pct, vol_multiplier, duration_days)
        parametric_scenarios = {
            "Mild Correction (-10%)":    (-0.10, 1.5,  21),   # ~1 month
            "Bear Market (-25%)":        (-0.25, 2.0,  63),   # ~3 months
            "Severe Crash (-40%)":       (-0.40, 3.0, 126),   # ~6 months (GFC-like)
            "Tail Event (-55%)":         (-0.55, 4.0, 252),   # ~1 year   (Dot-com-like)
        }

        parametric_results = {}
        np.random.seed(42)

        for name, (market_drop, vol_mult, duration) in parametric_scenarios.items():
            # Shocked daily mean: distribute total drop evenly + downward drift
            shocked_daily_mean = (market_drop / duration) + (daily_mean * 0.5)
            shocked_daily_vol = daily_vol * vol_mult

            # Simulate 500 paths through the shock window (Monte Carlo)
            n_paths = 500
            all_paths_dd = []
            all_paths_tr = []

            for _ in range(n_paths):
                sim_returns = np.random.normal(shocked_daily_mean, shocked_daily_vol, duration)
                cumulative = np.cumprod(1 + sim_returns)
                peak = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - peak) / peak
                all_paths_dd.append(float(drawdowns.min()))
                all_paths_tr.append(float(cumulative[-1] - 1.0))

            # Report median and 5th percentile (worst 5% of paths)
            parametric_results[name] = {
                "median_drawdown":    round(float(np.median(all_paths_dd)) * 100, 2),
                "p5_drawdown":        round(float(np.percentile(all_paths_dd, 5)) * 100, 2),
                "median_return":      round(float(np.median(all_paths_tr)) * 100, 2),
                "p5_return":          round(float(np.percentile(all_paths_tr, 5)) * 100, 2),
                "duration_days":      duration,
            }

        results["parametric"] = parametric_results
        return results

    def run_monte_carlo(self, prices_df, weights, n_simulations=500, horizon_days=252):
        """
        FORWARD-LOOKING MONTE CARLO SIMULATION
        ────────────────────────────────────────
        Simulates n_simulations possible future paths for the portfolio
        over horizon_days trading days using:
          - Expected return and volatility bootstrapped from historical data
          - Cholesky decomposition of the covariance matrix to preserve
            realistic correlations between assets (not just individual vols)

        This is NOT a prediction. It answers: "Given what we know about
        how these assets move and correlate, what is the distribution of
        possible outcomes over the next year?"

        Returns:
          - percentile paths (p5, p25, median, p75, p95) for charting
          - summary stats: expected return, VaR 95%, CVaR 95%, prob of loss
        """
        returns = prices_df.pct_change().dropna()
        n_assets = len(prices_df.columns)

        # Fit distribution parameters from history
        mean_returns = returns.mean().values          # daily mean per asset
        cov_matrix = returns.cov().values             # full covariance matrix

        # Cholesky decomposition — preserves asset correlations in simulation
        try:
            chol = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback if matrix not positive definite: add small regularization
            cov_matrix += np.eye(n_assets) * 1e-8
            chol = np.linalg.cholesky(cov_matrix)

        np.random.seed(42)
        all_paths = []

        for _ in range(n_simulations):
            # Draw correlated random shocks
            z = np.random.standard_normal((horizon_days, n_assets))
            correlated_shocks = z @ chol.T

            # Daily returns = mean + correlated shock
            sim_daily_returns = mean_returns + correlated_shocks

            # Portfolio daily returns
            port_returns = sim_daily_returns @ weights

            # Cumulative growth path
            path = np.cumprod(1 + port_returns)
            all_paths.append(path)

        all_paths = np.array(all_paths)  # shape: (n_simulations, horizon_days)

        # Final value distribution
        final_values = all_paths[:, -1]

        # Percentile paths for fan chart
        percentiles = {
            "p5":     np.percentile(all_paths, 5,  axis=0).tolist(),
            "p25":    np.percentile(all_paths, 25, axis=0).tolist(),
            "median": np.percentile(all_paths, 50, axis=0).tolist(),
            "p75":    np.percentile(all_paths, 75, axis=0).tolist(),
            "p95":    np.percentile(all_paths, 95, axis=0).tolist(),
        }

        # Summary statistics
        expected_return = float(np.mean(final_values) - 1.0)
        var_95 = float(np.percentile(final_values, 5) - 1.0)
        cvar_95 = float(np.mean(final_values[final_values <= np.percentile(final_values, 5)]) - 1.0)
        prob_loss = float(np.mean(final_values < 1.0))
        prob_gain_10 = float(np.mean(final_values > 1.10))

        return {
            "paths": percentiles,
            "horizon_days": horizon_days,
            "n_simulations": n_simulations,
            "expected_return": round(expected_return * 100, 2),
            "var_95":          round(var_95 * 100, 2),
            "cvar_95":         round(cvar_95 * 100, 2),
            "prob_loss":       round(prob_loss * 100, 1),
            "prob_gain_10":    round(prob_gain_10 * 100, 1),
        }