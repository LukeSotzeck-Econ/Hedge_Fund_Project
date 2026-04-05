import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Standard path configuration for modular architecture
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import internal modular components
from engines.agent_hub.risk_manager.risk_core import RiskGatekeeper
from engines.agent_hub.quant_manager.quant_core import QuantModeler
from engines.data_nexus.nexus_core import DataNexus
from engines.valuation_vault.valuation_core import ActiveBusinessModel, PositionSizer
from engines.agent_hub.agent_core import MacroAgent, QualiAgent, RiskAgent, QuantAgent

# UI Configuration
st.set_page_config(page_title="Horizon Sentinel | Portfolio Intelligence", layout="wide")

st.title("Horizon Sentinel")
st.markdown("#### Portfolio Orchestration and Deep Dive Diagnostic Suite")
st.markdown("---")

# --- SIDEBAR: THE SLUICE (Input Management) ---
st.sidebar.header("Order Entry | Asset Allocation")
default_book = pd.DataFrame({
    "Ticker": ["JPM", "AAPL"], 
    "Exposure ($)": [60000.0, 30000.0]
})
edited_book = st.sidebar.data_editor(default_book, num_rows="dynamic", width="stretch", hide_index=True)

st.sidebar.markdown("---")
proposed_ticker = st.sidebar.text_input("Proposed Asset (Ticker)", value="TSLA").upper()
proposed_exposure = st.sidebar.number_input("Proposed Allocation ($)", value=10000.0, step=1000.0)

execute_trade = st.sidebar.button("Execute Quantitative Analysis", type="primary")

# --- DATA PERSISTENCE LAYER ---
if execute_trade:
    current_book = {row["Ticker"].upper(): float(row["Exposure ($)"]) for _, row in edited_book.iterrows() if row["Ticker"]}
    benchmark = "SPY"
    universe = list(set(list(current_book.keys()) + [proposed_ticker] + [benchmark]))
    
    with st.status("Initializing Multi-Agent Logic...", expanded=True) as status:
        # 1. DATA NEXUS: Centralized data ingestion
        st.write("Fetching historical and fundamental data via Nexus...")
        nexus = DataNexus(universe)
        prices, fundamentals = nexus.fetch_everything(period="5y")
        
        # 2. DETERMINISTIC ENGINES: Risk and Quantitative modeling
        st.write("Calculating Marginal Risk (CVaR)...")
        gatekeeper = RiskGatekeeper(confidence_level=0.95)
        risk_impact = gatekeeper.evaluate_trade_impact(current_book, proposed_ticker, proposed_exposure, prices)
        
        st.write("Modeling Portfolio Efficiency...")
        quant = QuantModeler(risk_free_rate=0.04)
        quant_impact = quant.evaluate_trade_impact(current_book, proposed_ticker, proposed_exposure, prices, prices[benchmark])
        
        # 3. OPTIMIZATION: Solving for MSR (Maximum Sharpe Ratio)
        st.write("Optimizing Portfolio Weights...")
        opt_universe = [t for t in universe if t != benchmark]
        opt_weights_dict = quant.optimize_portfolio(prices[opt_universe])
        
        # 4. BACKTEST: Historical reconstruction
        st.write("Running backtest series...")
        new_holdings = current_book.copy()
        new_holdings[proposed_ticker] = new_holdings.get(proposed_ticker, 0.0) + proposed_exposure
        new_total = sum(new_holdings.values())
        new_weights = np.array([new_holdings[t] / new_total for t in new_holdings.keys()])
        
        equity_curve = quant.get_backtest_series(prices[list(new_holdings.keys())], new_weights)
        bench_curve = quant.get_backtest_series(prices[[benchmark]], [1.0])

        # 5. CORRELATION MATRIX
        st.write("Computing correlation matrix...")
        corr_matrix = quant.get_correlation_matrix(prices[opt_universe])

        # 6. STRESS TEST (historical + parametric)
        st.write("Running stress scenarios...")
        stress_results = quant.run_stress_test(prices[list(new_holdings.keys())], new_weights)

        # 7. MONTE CARLO FORWARD SIMULATION
        st.write("Running Monte Carlo forward simulation (500 paths)...")
        mc_results = quant.run_monte_carlo(prices[list(new_holdings.keys())], new_weights)

        # Save all results to the Streamlit Session State memory
        st.session_state['analysis_data'] = {
            'proposed_ticker': proposed_ticker,
            'prices': prices,
            'fundamentals': fundamentals,
            'risk_impact': risk_impact,
            'quant_impact': quant_impact,
            'opt_universe': opt_universe,
            'opt_weights_dict': opt_weights_dict,
            'equity_curve': equity_curve,
            'bench_curve': bench_curve,
            'benchmark': benchmark,
            'corr_matrix': corr_matrix,
            'stress_results': stress_results,
            'mc_results': mc_results
        }
        
        status.update(label="Analysis Complete.", state="complete", expanded=False)

# --- RENDERING LAYER ---
# Only render the dashboard if the data exists in memory
if 'analysis_data' in st.session_state:
    data = st.session_state['analysis_data']
    
    # Re-instantiate local variables from memory
    proposed_ticker = data['proposed_ticker']
    prices = data['prices']
    fundamentals = data['fundamentals']
    risk_impact = data['risk_impact']
    quant_impact = data['quant_impact']
    opt_universe = data['opt_universe']
    opt_weights_dict = data['opt_weights_dict']
    equity_curve = data['equity_curve']
    bench_curve = data['bench_curve']
    benchmark = data['benchmark']
    corr_matrix = data['corr_matrix']
    stress_results = data['stress_results']
    mc_results = data['mc_results']

    # Re-instantiate tools for rendering logic
    gatekeeper = RiskGatekeeper(confidence_level=0.95)
    quant = QuantModeler(risk_free_rate=0.04)

    # --- ROW 1: CORE PORTFOLIO METRICS ---
    st.subheader(f"Portfolio Impact Report: {proposed_ticker}")
    is_vetoed, risk_msg = gatekeeper.veto_check(risk_impact['new_cvar_pct'], max_tolerable_loss=-0.08)
    if is_vetoed: st.error(risk_msg)
    else: st.success(risk_msg)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Gross Book Value", f"${risk_impact['new_portfolio_value']:,.2f}")
    m2.metric("Portfolio CVaR", f"{risk_impact['new_cvar_pct']*100:.2f}%", 
              delta=f"{risk_impact['marginal_cvar_pct']*100:.2f}%", delta_color="inverse")
    m3.metric("Annualized Return", f"{quant_impact['new_return']*100:.2f}%", 
              delta=f"{quant_impact['delta_return']*100:.2f}%")
    m4.metric("Portfolio Sharpe", f"{quant_impact['new_sharpe']:.2f}", 
              delta=f"{quant_impact['delta_sharpe']:.2f}")
    m5.metric("Max Drawdown", f"{quant_impact['new_mdd']*100:.2f}%", 
              delta=f"{quant_impact['delta_mdd']*100:.2f}%", delta_color="inverse")

    st.markdown("---")

    # --- ROW 2: EQUITY CURVE VISUALIZATION ---
    st.markdown("#### Cumulative Performance: Proposed Portfolio vs. S&P 500")
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values * 100, name="Proposed Book", line=dict(color="#00FFAA", width=3)))
    fig_backtest.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values * 100, name="S&P 500 Benchmark", line=dict(color="#888888", dash='dash')))
    fig_backtest.update_layout(yaxis_title="Normalized Growth (%)", hovermode="x unified", template="plotly_dark", height=400)
    st.plotly_chart(fig_backtest, use_container_width=True)

    st.markdown("---")

    # --- ROW 3: QUANTITATIVE MAP AND OPTIMIZATION ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Risk/Return Efficiency Map")
        plot_data = []
        for ticker in opt_universe:
            m = quant.evaluate_asset(prices[ticker], prices[benchmark])
            plot_data.append({"Ticker": ticker, "Return (%)": m['annualized_return']*100, "Volatility (%)": m['annualized_volatility']*100, "Sharpe": m['sharpe_ratio'], "Type": "Asset"})
        
        plot_data.append({"Ticker": "CURRENT STATE", "Return (%)": quant_impact['current_return']*100, "Volatility (%)": quant_impact['current_vol']*100, "Sharpe": quant_impact['current_sharpe'], "Type": "Portfolio"})
        plot_data.append({"Ticker": "PROPOSED STATE", "Return (%)": quant_impact['new_return']*100, "Volatility (%)": quant_impact['new_vol']*100, "Sharpe": quant_impact['new_sharpe'], "Type": "Portfolio"})
        
        df_plot = pd.DataFrame(plot_data)
        sizes = [16 if t.endswith("STATE") else 8 for t in df_plot["Ticker"]]

        fig_scatter = px.scatter(df_plot, x="Volatility (%)", y="Return (%)", text="Ticker", color="Sharpe", symbol="Type", size=sizes, size_max=16, color_continuous_scale="Viridis")
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=40), template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("#### Optimal Allocation (MSR)")
        st.caption("Calculated weights for maximum risk-adjusted return")
        for t, w in opt_weights_dict.items():
            st.progress(float(w), text=f"{t}: {w*100:.1f}%")

        st.markdown("---")
        st.markdown("#### Asset Correlation Matrix")
        st.caption("Values near 1.0 = assets move together (less diversification)")
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}", showscale=True
        ))
        fig_corr.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0), template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # --- ROW 4: STRESS TEST ---
    st.markdown("#### Stress Test")
    tab_hist, tab_param, tab_mc = st.tabs([
        "Historical Scenarios", "Parametric Shocks", "Monte Carlo Forecast"
    ])

    with tab_hist:
        st.caption("Portfolio replayed through actual crisis windows — only available if prices cover that period")
        hist = stress_results["historical"]
        h_cols = st.columns(4)
        icons = ["🦠", "📈", "🏦", "💻"]
        for col, (name, res), icon in zip(h_cols, hist.items(), icons):
            short = name.split("(")[0].strip()
            with col:
                if res["max_drawdown"] is not None:
                    st.metric(f"{icon} {short}", f"{res['max_drawdown']:.1f}% MDD",
                              delta=f"{res['total_return']:.1f}% return", delta_color="normal")
                    st.caption(f"{res['trading_days']} days · Historical")
                else:
                    st.metric(f"{icon} {short}", "N/A")
                    st.caption(res.get("note", "No data"))

    with tab_param:
        st.caption("Parametric shocks calibrated to crisis magnitudes — works regardless of data window. Shows median and worst-5% path across 500 simulations.")
        param = stress_results["parametric"]
        p_cols = st.columns(4)
        for col, (name, res) in zip(p_cols, param.items()):
            with col:
                st.metric(name, f"{res['median_drawdown']:.1f}% MDD",
                          delta=f"{res['median_return']:.1f}% return", delta_color="normal")
                st.caption(f"Worst 5%: {res['p5_drawdown']:.1f}% MDD · {res['duration_days']}d")

    with tab_mc:
        st.caption(f"Forward simulation: {mc_results['n_simulations']} paths · {mc_results['horizon_days']} trading days (~1 year) · Cholesky-correlated returns")

        days = list(range(1, mc_results['horizon_days'] + 1))
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=days + days[::-1],
            y=[v*100 for v in mc_results['paths']['p95']] + [v*100 for v in mc_results['paths']['p5'][::-1]],
            fill='toself', fillcolor='rgba(0,255,170,0.08)',
            line=dict(color='rgba(255,255,255,0)'), name='5–95th percentile', showlegend=True
        ))
        fig_mc.add_trace(go.Scatter(
            x=days + days[::-1],
            y=[v*100 for v in mc_results['paths']['p75']] + [v*100 for v in mc_results['paths']['p25'][::-1]],
            fill='toself', fillcolor='rgba(0,255,170,0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='25–75th percentile', showlegend=True
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=[v*100 for v in mc_results['paths']['median']],
            line=dict(color='#00FFAA', width=2.5), name='Median path'
        ))
        fig_mc.add_hline(y=100, line_dash="dash", line_color="#888888", annotation_text="Break-even")
        fig_mc.update_layout(
            yaxis_title="Portfolio Value (start = 100)",
            xaxis_title="Trading Days",
            template="plotly_dark", height=380, hovermode="x unified"
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Expected Return", f"{mc_results['expected_return']:+.1f}%")
        s2.metric("VaR 95%", f"{mc_results['var_95']:.1f}%")
        s3.metric("CVaR 95%", f"{mc_results['cvar_95']:.1f}%")
        s4.metric("Prob. of Loss", f"{mc_results['prob_loss']:.0f}%")
        s5.metric("Prob. >+10%", f"{mc_results['prob_gain_10']:.0f}%")

    # --- ROW 5: AGENT-BASED DEEP DIVE ---
    st.markdown("---")
    st.subheader(f"Qualitative Intelligence: {proposed_ticker}")

    if st.button(f"Initialize Board of Directors Meeting for {proposed_ticker}"):
        with st.spinner("Processing agent heuristics and valuation modeling..."):
            macro = MacroAgent()
            quali = QualiAgent()
            risk_agent = RiskAgent()
            quant_agent = QuantAgent()

            regime = macro.get_market_regime()
            q_score = quali.analyze_company(proposed_ticker)
            r_scout = risk_agent.scan_for_risks(proposed_ticker)

            ticker_fundamentals = fundamentals.get(proposed_ticker, {})
            quanti_est = quant_agent.get_forward_estimates(proposed_ticker, ticker_fundamentals)

            abm = ActiveBusinessModel(
                ticker=proposed_ticker,
                sector=ticker_fundamentals.get('sector', 'Technology'),
                macro_sentiment=regime['macro_sentiment']
            )
            report = abm.run_master_engine(q_score, quanti_est, r_scout)

            # Position Sizing
            sizer = PositionSizer()
            asset_vol = None
            try:
                asset_vol = quant.evaluate_asset(
                    prices[proposed_ticker], prices[benchmark]
                )['annualized_volatility']
            except Exception:
                pass

            sizing = sizer.calculate(
                abm_score=report['S_Netto (Final Score)'],
                portfolio_value=risk_impact['new_portfolio_value'],
                current_cvar_pct=risk_impact['new_cvar_pct'],
                asset_volatility=asset_vol
            )

            # Board Verdict
            st.markdown("#### Board Verdict")
            action = sizing['action']
            if "STRONG BUY" in action or action == "BUY":
                st.success(f"**{action}** — {sizing['conviction']}")
            elif "HOLD" in action:
                st.warning(f"**{action}** — {sizing['conviction']}")
            else:
                st.error(f"**{action}** — {sizing['conviction']}")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("ABM Score", f"{sizing['abm_score']} / 10")
            p2.metric("Recommended Weight", f"{sizing['recommended_weight']}%")
            p3.metric("Position Size", f"${sizing['recommended_dollar']:,.0f}")
            p4.metric("Est. Win Rate", f"{sizing['win_rate_est']}%")
            st.caption(
                f"Half-Kelly: {sizing['kelly_half_pct']}%  ·  "
                f"Full Kelly: {sizing['kelly_full_pct']}%  ·  "
                f"Conviction edge: {sizing['edge']}%"
            )

            st.markdown("---")
            st.markdown("#### Quantitative DNA Matrix")
            dna = report['Quanti_DNA']
            dna_cols = st.columns(len(dna))
            for col, (pillar, score) in zip(dna_cols, dna.items()):
                col.metric(pillar, f"{score:.1f} / 10")

            st.markdown("#### Strategy Details")
            d1, d2 = st.columns(2)
            d1.info(f"**Classification**\n\n{report['Identity']}")
            d2.info(f"**Macro-adjusted Weights**\n\n{report['Weights']}")