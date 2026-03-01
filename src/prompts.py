"""Prompt templates and market classification for AI analyzers."""

# ---------------------------------------------------------------------------
# Analysis Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are an expert prediction market trader. Your job is to find mispriced markets and bet on them.

## Market
- **Question**: {question}
- **Description**: {description}
- **Outcomes**: {outcomes}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Spread**: {spread}
- **Volume**: ${volume}
- **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
- **End date**: {end_date}
{price_history_section}
{momentum_section}
{order_book_section}
{related_markets_section}
{market_maturity_section}
{web_context_section}
{category_instructions}

## Instructions
1. Based on your knowledge of current events, historical patterns, and base rates, estimate the TRUE probability that the YES outcome occurs.
2. Compare your estimate to the market price.
3. If you have an edge, recommend a bet. If not, SKIP.

## Field Definitions
- **estimated_probability**: Your honest best estimate of the TRUE probability YES occurs [0.0-1.0]. This drives bet sizing — be precise.
- **confidence**: How confident you are in your edge over the market [0.0-1.0]. High confidence = you believe the market is clearly wrong and you have strong evidence. Low confidence = your edge is uncertain or based on weak signals. This scales position size.
- **reasoning**: 2-3 sentence summary of your key insight and why the market may be mispriced.

## Important
- Be calibrated. If you're unsure, your probability should reflect that uncertainty.
- Consider what information the market already has priced in.
- Think about base rates for similar events.
- A 1% market doesn't mean free money — most things priced at 1% really are unlikely.
- Only bet when you have a genuine informational or analytical edge.
- Beware of recency bias: recent price moves do not equal fundamental change.
- If your estimate is >85% or <15%, double-check — extreme probabilities are often overconfident.

Respond with ONLY a valid JSON object, no text before or after:
{{"recommendation": "BUY_YES" or "BUY_NO" or "SKIP", "confidence": 0.0 to 1.0, "estimated_probability": 0.0 to 1.0, "reasoning": "2-3 sentence analysis"}}"""

COT_STEP1_PROMPT = """You are an expert prediction market analyst.

## Market
- **Question**: {question}
- **Description**: {description}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Volume**: ${volume} | **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
{price_history_section}
{momentum_section}
{order_book_section}
{related_markets_section}
{market_maturity_section}
{web_context_section}
{category_instructions}

## Task: Argue FOR YES
Build the STRONGEST possible case that YES will occur. Consider:
- What evidence or trends support YES?
- What base rates or historical precedents favor YES?
- What would have to be true for YES to win?
- What is the upper bound of a reasonable YES probability?

Write 3-5 concise bullet points arguing FOR YES. Be specific and evidence-based."""

COT_STEP2_PROMPT = """You are an expert prediction market analyst.

## Market
- **Question**: {question}
- **Description**: {description}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Time remaining**: {time_remaining}
{price_history_section}
{momentum_section}
{order_book_section}
{related_markets_section}
{market_maturity_section}
{web_context_section}
{category_instructions}

## The Case FOR YES:
{yes_argument}

## Task: Argue FOR NO
Now build the STRONGEST possible case that NO will occur. Consider:
- What evidence or trends support NO?
- What are the base rates for events like this failing?
- What flaws exist in the YES argument above?
- What is the lower bound of a reasonable YES probability?

Write 3-5 concise bullet points arguing FOR NO. Be specific and evidence-based."""

COT_STEP3_PROMPT = """You are an expert prediction market trader. Your job is to find mispriced markets.

## Market
- **Question**: {question}
- **Description**: {description}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Volume**: ${volume} | **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
{momentum_section}
{order_book_section}
{related_markets_section}
{market_maturity_section}
{web_context_section}
{category_instructions}

## Case FOR YES:
{yes_argument}

## Case FOR NO:
{no_argument}

## Task: Synthesize and Decide
Weigh both arguments carefully. Consider:
- Which case is stronger, and by how much?
- Does the market price already reflect the consensus view?
- Do you have a genuine informational or analytical edge?
- Only bet when your estimate differs meaningfully from the market price.

Respond with ONLY valid JSON:
{{"recommendation": "BUY_YES" or "BUY_NO" or "SKIP", "confidence": 0.0 to 1.0, "estimated_probability": 0.0 to 1.0, "reasoning": "your synthesis in 2-3 sentences"}}"""

# ---------------------------------------------------------------------------
# Debate Prompts
# ---------------------------------------------------------------------------

DEBATE_REBUTTAL_PROMPT = """You are an expert prediction market analyst engaged in a structured debate.

## Market
- **Question**: {question}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Volume**: ${volume} | **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
{web_context_section}

## Your Original Analysis ({own_model})
- Recommendation: {own_recommendation}
- Estimated probability: {own_probability:.1%}
- Confidence: {own_confidence:.0%}
- Reasoning: {own_reasoning}

## Other Models' Analyses
{other_analyses}

## Task: Rebuttal Round
1. Identify the STRONGEST opposing argument from other models.
2. Identify BLIND SPOTS in opponents' reasoning — what are they missing?
3. Consider whether any opposing argument should change your position.
4. Either UPDATE your position with justification or HOLD with strengthened reasoning.

Respond with ONLY valid JSON:
{{"updated_recommendation": "BUY_YES" or "BUY_NO" or "SKIP", "updated_probability": 0.0 to 1.0, "updated_confidence": 0.0 to 1.0, "rebuttal_reasoning": "your rebuttal and updated analysis"}}"""

DEBATE_SYNTHESIS_PROMPT = """You are a master synthesizer resolving a multi-model prediction market debate.

## Market
- **Question**: {question}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Volume**: ${volume} | **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
{web_context_section}

## Round 1: Independent Analyses
{round1_analyses}

## Round 2: Rebuttals
{round2_rebuttals}

## Task: Final Synthesis
Weigh all arguments and rebuttals to produce a final decision:
1. Which arguments SURVIVED the debate strongest?
2. Did any model change position for a GOOD reason (new info vs capitulation)?
3. Where did models CONVERGE (strong signal) vs DIVERGE (genuine uncertainty)?
4. What is the calibrated final probability accounting for all perspectives?

Respond with ONLY valid JSON:
{{"recommendation": "BUY_YES" or "BUY_NO" or "SKIP", "confidence": 0.0 to 1.0, "estimated_probability": 0.0 to 1.0, "reasoning": "synthesis of debate in 2-3 sentences"}}"""

# ---------------------------------------------------------------------------
# Market Classification
# ---------------------------------------------------------------------------

MARKET_CATEGORIES = {
    "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "defi",
               "blockchain", "token", "coinbase", "binance", "altcoin", "memecoin",
               "dogecoin"],
    "politics": ["president", "election", "trump", "congress", "senate", "governor",
                 "democrat", "republican", "vote", "ballot", "poll", "impeach",
                 "supreme court", "legislation", "bill", "veto", "biden", "desantis",
                 "party", "mayor"],
    "sports": ["game", "match", "team", "championship", "league", "playoff",
               "winner", " vs ", "nfl", "nba", "mlb", "nhl", "world cup", "super bowl",
               "tournament", "season", "points", "goals", "o/u", "ufc", "soccer",
               "football", "basketball", "baseball"],
    "economics": ["stock", "s&p", "nasdaq", "dow", "fed", "interest rate",
                  "gdp", "inflation", "recession", "treasury", "market cap", "ipo",
                  "unemployment", "cpi", "jobs", "tariff"],
    "tech": [" ai ", "artificial intelligence", "openai", "google", "apple", "tesla",
             "spacex", "launch", "ipo", "acquisition", "gpt", "nvidia", "microsoft",
             "release"],
    "science": ["fda", "approval", "vaccine", "trial", "study", "nasa", "climate",
                "earthquake", "hurricane", "drug"],
    "entertainment": ["oscar", "grammy", "emmy", "movie", "album", "box office",
                      "streaming", "netflix", "disney"],
    "geopolitics": ["war", "ukraine", "russia", "china", "nato", "sanctions",
                    "ceasefire", "nuclear", "military"],
}

CATEGORY_INSTRUCTIONS = {
    "crypto": """
## Category Guidance: Crypto
- Focus on technical indicators: momentum, support/resistance levels, recent price action
- Consider macro conditions: Fed policy, dollar strength, risk-on/risk-off sentiment
- Account for crypto's high volatility — wide probability intervals are appropriate
- Base rates: crypto price targets are often missed in both directions""",

    "politics": """
## Category Guidance: Politics
- Weight polling averages carefully — consider historical polling bias
- Apply base rates: incumbents win X% of the time, etc.
- Consider structural factors over individual events
- Recent news often causes overreaction — revert to base rates unless change is structural""",

    "sports": """
## Category Guidance: Sports
- Focus on recent form (last 5-10 games), not just season averages
- Head-to-head records and home/away splits matter
- Account for injury reports and rest days
- Bookmaker lines are highly efficient — require strong evidence to bet against them""",

    "economics": """
## Category Guidance: Economics
- Examine macro indicators and Fed policy trajectory
- Consider consensus forecasts and how often they prove accurate
- Look at historical base rates for similar economic milestones
- Factor in uncertainty already priced into related assets""",

    "tech": """
## Category Guidance: Technology
- For product launches: company track record matters most
- For AI milestones: progress is faster than expected but slower than press suggests
- Distinguish official announcements from speculation
- Consider competitive dynamics and market positioning""",

    "science": """
## Category Guidance: Science
- For regulatory/FDA: use historical approval rates by stage
- For clinical trials: phase success rates are well-documented base rates
- For space launches: check mission history and weather constraints
- Distinguish peer-reviewed findings from pre-prints and press coverage""",

    "entertainment": "",

    "geopolitics": "",

    "general": "",
}


def classify_market(question: str) -> str:
    """Classify a market question into a category by keyword matching."""
    q = question.lower()
    for category in MARKET_CATEGORIES:
        if any(kw in q for kw in MARKET_CATEGORIES[category]):
            return category
    return "general"
