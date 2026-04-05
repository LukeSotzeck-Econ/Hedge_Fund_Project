from engines.data_nexus.nexus_core import DataNexus
from engines.valuation_vault.valuation_core import ActiveBusinessModel
from engines.agent_hub.agent_core import MacroAgent, QualiAgent, RiskAgent, QuantAgent

class PortfolioManager:
    """
    The Orchestrator. Connects the Data Layer, Agent Hub, and Math Engine.
    """
    def __init__(self, tickers):
        self.tickers = tickers
        # Initialize Data Pipeline
        self.nexus = DataNexus(tickers)
        
        # Initialize the Board of Directors (Agents)
        self.macro_agent = MacroAgent()
        self.quali_agent = QualiAgent()
        self.risk_agent = RiskAgent()
        self.quant_agent = QuantAgent()

    def run_analysis(self):
        print("🚀 Starting Multi-Asset Intelligence System...\n")
        
        # 1. Fetch raw data from the Nexus (Kraftwerk)
        prices, fundamentals = self.nexus.fetch_everything(period="1y")
        
        # 2. Get Global Market Regime ONCE for the whole portfolio
        regime_data = self.macro_agent.get_market_regime()
        current_sentiment = regime_data.get('macro_sentiment', 0.0)
        
        results = {}
        for ticker in self.tickers:
            print(f"\n{'='*40}")
            print(f"🏢 ASSEMBLE BOARD FOR: {ticker}")
            print(f"{'='*40}")
            
            # 3. Agents deliver their reports
            quali_inputs = self.quali_agent.analyze_company(ticker)
            risks = self.risk_agent.scan_for_risks(ticker)
            
            stock_data = fundamentals.get(ticker, {})
            quanti_inputs = self.quant_agent.get_forward_estimates(ticker, stock_data)
            
            # 4. Feed all intelligence into the Math Engine (Werkstatt)
            print(f"⚙️ [Math Engine] Processing ABM V6.0 for {ticker}...")
            sector = stock_data.get('sector', 'Technology')
            
            abm = ActiveBusinessModel(
                ticker=ticker, 
                sector=sector, 
                strategy_type='QG', 
                macro_sentiment=current_sentiment
            )
            
            final_report = abm.run_master_engine(quali_inputs, quanti_inputs, risks)
            results[ticker] = final_report
            
            # 5. The Final Verdict
            print(f"\n📊 FINAL DIAGNOSIS: {ticker}")
            print(f"   Identity: {final_report['Identity']}")
            print(f"   Weights : {final_report['Weights']}")
            print(f"   Score   : {final_report['S_Netto (Final Score)']}/10")
            print(f"   DNA     : {final_report['Quanti_DNA']}")

if __name__ == "__main__":
    # Running the full simulated board meeting for two different stocks
    manager = PortfolioManager(['AAPL', 'JPM'])
    manager.run_analysis()