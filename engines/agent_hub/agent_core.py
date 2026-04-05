import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GroqAgent:
    """
    Base Agent using Groq API with llama-3.3-70b.
    - No rate limit issues on free tier (30 RPM, 14400 RPD)
    - response_format={"type": "json_object"} forces pure JSON output
    - Sub-second responses (Groq runs on custom LPU hardware)
    """
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("WARNING: No GROQ_API_KEY found — running in Simulation Mode.")
            self.simulation_mode = True
        else:
            self.client = Groq(api_key=api_key)
            self.model_id = "llama-3.3-70b-versatile"
            self.simulation_mode = False

    def _ask_groq(self, system_prompt, user_prompt):
        """
        Single clean call with forced JSON mode.
        Groq's json_object mode guarantees valid JSON — no parsing gymnastics needed.
        System prompt defines the schema, user prompt provides the context.
        """
        if self.simulation_mode:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"Groq Agent Error: {e}")
            return None


class MacroAgent(GroqAgent):
    def get_market_regime(self):
        print("[Macro Agent] Analyzing global market sentiment...")
        system = (
            'You are a macro economist. Respond ONLY with a JSON object. '
            'Schema: {"macro_sentiment": <float>} '
            'where macro_sentiment is between -1.0 (extreme panic/recession) '
            'and 1.0 (euphoria/boom).'
        )
        user = "What is the current global macro sentiment based on recent market conditions?"
        res = self._ask_groq(system, user)
        return res if res else {"macro_sentiment": 0.0}


class QualiAgent(GroqAgent):
    def analyze_company(self, ticker):
        print(f"[Quali Agent] Qualitative diagnostic for {ticker}...")
        system = (
            'You are an equity research analyst. Respond ONLY with a JSON object. '
            'Schema: {"moat": <float>, "mgmt": <float>, "market": <float>} '
            'All values between 0 (very weak) and 10 (exceptional). '
            'moat = competitive moat strength. '
            'mgmt = management quality and capital allocation track record. '
            'market = size and growth potential of the addressable market.'
        )
        user = f"Evaluate {ticker} on moat, management quality, and market opportunity."
        res = self._ask_groq(system, user)
        return res if res else {'moat': 5.0, 'mgmt': 5.0, 'market': 5.0}


class RiskAgent(GroqAgent):
    def scan_for_risks(self, ticker):
        print(f"[Risk Agent] Scanning risks for {ticker}...")
        system = (
            'You are a risk analyst. Respond ONLY with a JSON object. '
            'Schema: {"extreme_insolvency": <bool>, "high_debt_reg": <bool>, '
            '"concentration": <bool>, "industry_risk": <bool>} '
            'extreme_insolvency: true only if bankruptcy risk is imminent. '
            'high_debt_reg: true if debt load or regulatory fines are a serious threat. '
            'concentration: true if revenue is dangerously concentrated in one customer or region. '
            'industry_risk: true if the sector faces major structural headwinds.'
        )
        user = f"Assess the key risk flags for {ticker} as of today."
        res = self._ask_groq(system, user)
        return res if res else {
            'extreme_insolvency': False, 'high_debt_reg': False,
            'concentration': False, 'industry_risk': False
        }


class QuantAgent(GroqAgent):
    def get_forward_estimates(self, ticker, fundamental_data):
        print(f"[Quant Agent] Forward estimates for {ticker}...")
        pe_forward = fundamental_data.get('pe_ratio', 15) or 15
        system = (
            'You are a sell-side quantitative analyst. Respond ONLY with a JSON object. '
            f'Schema: {{"cagr": <float>, "roic": <float>, "wacc": <float>, '
            f'"net_debt_ebitda": <float>, "dcf_upside": <float>, "peg": <float>, '
            f'"pe_current": <float>, "pe_hist_5y": <float>}} '
            'cagr: 3-year forward revenue CAGR as decimal (e.g. 0.12 = 12%). '
            'roic: return on invested capital as decimal. '
            'wacc: weighted average cost of capital as decimal. '
            'net_debt_ebitda: net debt divided by EBITDA (leverage ratio). '
            'dcf_upside: DCF intrinsic value vs current price as decimal '
            '(positive = undervalued, negative = overvalued). '
            'peg: price-to-earnings divided by growth rate. '
            'pe_current: current forward P/E ratio as float. '
            'pe_hist_5y: 5-year historical average P/E as float.'
        )
        user = (
            f"Provide forward-looking financial estimates for {ticker}. "
            f"The current forward P/E from market data is {pe_forward}. "
            f"Use this as pe_current and estimate the rest based on your knowledge."
        )
        res = self._ask_groq(system, user)
        return res if res else {
            'cagr': 0.05, 'roic': 0.10, 'wacc': 0.08, 'net_debt_ebitda': 2.0,
            'dcf_upside': 0.0, 'peg': 2.0, 'pe_current': float(pe_forward), 'pe_hist_5y': 15.0
        }