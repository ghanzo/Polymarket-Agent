# Academic Research: Prediction Markets & LLM Forecasting

> Last updated: 2026-02-26

## LLM Forecasting

### "LLMs Can Teach Themselves to Better Predict the Future"
- **Source:** [arXiv 2502.05253](https://arxiv.org/abs/2502.05253)
- **Date:** 2025
- **Key finding:** Outcome-driven fine-tuning framework using 12,100 Polymarket binary questions. Used model self-play to generate reasoning trajectories, then DPO fine-tuning. **Improved prediction accuracy of 14B parameter models by 7-10%**, bringing them on par with GPT-4o.
- **Relevance to us:** Directly applicable — we could fine-tune smaller models to match larger ones, dramatically reducing API costs while maintaining accuracy.
- **Status:** [UNVERIFIED] — need to replicate their methodology

### "Going All-In on LLM Accuracy: Fake Prediction Markets, Real Confidence Signals"
- **Source:** [arXiv 2512.05998](https://arxiv.org/abs/2512.05998)
- **Date:** 2025
- **Key finding:** Framing LLM evaluation as a betting game surfaces calibrated confidence signals. **Large bets (40,000+ coins) were correct ~99% of the time**, while small bets (<1,000 coins) showed only ~74% accuracy.
- **Relevance to us:** We should modify our prompts to ask models to "bet" on their predictions. The bet size becomes a calibration signal — high-bet predictions should be weighted more heavily in the ensemble.
- **Status:** [UNVERIFIED] — worth testing in our prompt engineering

### "Evaluating LLMs on Real-World Forecasting Against Expert Forecasters"
- **Source:** [arXiv 2507.04562](https://arxiv.org/abs/2507.04562)
- **Date:** 2025
- **Key finding:** LLMs have surpassed human crowd-level forecasting but still underperform expert forecaster groups.
- **Relevance to us:** Validates multi-model ensemble approach. Suggests combining LLM predictions with domain expert heuristics could close the gap.
- **Status:** Informational

### "Approaching Human-Level Forecasting with Language Models"
- **Source:** [arXiv 2402.18563](https://arxiv.org/abs/2402.18563)
- **Date:** 2024
- **Key finding:** Combining market probability with LLM predictions via a **mixture approach** improves predictive performance. LLMs act as complementary refinements to market predictions, not replacements.
- **Relevance to us:** Our analyzer should treat market price as a strong prior and only bet when the LLM has a confident divergence, not treat the LLM prediction as standalone truth.
- **Status:** Informational — aligns with our architecture

---

## Market Microstructure & Arbitrage

### "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
- **Source:** [arXiv 2508.03474](https://arxiv.org/abs/2508.03474)
- **Date:** 2025
- **Key finding:** Analyzed Polymarket 2024-2025 and found ~$40 million in realized arbitrage profit. Two types:
  - **Intra-exchange:** Mispricing within single markets where outcome prices don't sum to $1
  - **Inter-exchange:** Cross-platform price differences (Polymarket vs Kalshi)
- **Current state:** Average arbitrage window is now ~2.7 seconds (down from 12.3s in 2024). 73% of profits captured by sub-100ms bots.
- **Relevance to us:** Pure arbitrage is not viable without HFT infrastructure. But monitoring for occasional large mispricings (>5%) with longer windows could still work.
- **Status:** Informational — arbitrage not our primary strategy

### Kalshi Prediction Market Economics (UCD Working Paper 2025)
- **Source:** [UCD Economics WP2025_19](https://www.ucd.ie/economics/t4media/WP2025_19.pdf)
- **Date:** 2025
- **Key finding:** Kalshi prices display systematic **favorite-longshot bias**:
  - Contracts priced below 10 cents lose over 60% of their money
  - Contracts priced above 50 cents earn small positive returns
- **Relevance to us:** Strong evidence for longshot bias exploitation strategy. We should systematically avoid/short contracts < $0.10 and favor contracts > $0.50.
- **Status:** Informational — supports longshot bias strategy

---

## Calibration & Decision Making

### General calibration research (aggregated findings)
- Superforecasters average Brier score of ~0.15 (best humans)
- Prediction markets typically achieve Brier scores of 0.18-0.22
- LLMs (GPT-4 class) achieve Brier scores of 0.20-0.25 on forecasting tasks
- Ensemble of 3+ models typically improves Brier by 0.02-0.04
- **Key insight:** The gap between LLMs and markets is small (0.02-0.05). Profitable trading requires finding the subset of markets where LLMs have genuine informational edge, not betting on everything.

---

## Research Backlog

- [ ] Read full text of arXiv 2502.05253 (self-improving LLMs) — could we apply DPO to our prompt strategy?
- [ ] Read full text of arXiv 2512.05998 (betting-framed prompts) — implement "bet sizing" in our prompts
- [ ] Search for papers on optimal ensemble weighting for forecasting
- [ ] Search for papers on prediction market fee impact on Kelly criterion
- [ ] Look for research on category-specific forecasting accuracy (politics vs crypto vs sports)
