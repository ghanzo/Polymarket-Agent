"""ML Pre-Screening Funnel — Zero-cost feature extraction + lightweight model.

Filters 80-90% of markets before expensive LLM analysis.
Inspired by NavnoorBawa/polymarket-prediction-system which achieves
87-91% accuracy at 80%+ confidence using 52 text+metadata features.

Architecture:
    Gamma API (free) → extract features → score with ML model → top 10-20% → LLM

The model is trained on our own analysis_log + bets tables:
    Positive examples: markets where LLM found edge and bet was profitable
    Negative examples: markets LLM skipped or bet was unprofitable
"""

import logging
import math
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

from src.config import config
from src.models import Market
from src.prompts import MARKET_CATEGORIES as CATEGORY_KEYWORDS

logger = logging.getLogger("prescreener")


def extract_features(market: Market) -> dict[str, float]:
    """Extract features from a Market object — zero API cost.

    All inputs come from the Gamma API listing response (already fetched by scanner).
    Returns a dict of feature_name -> float value.
    """
    features: dict[str, float] = {}
    q = market.question.lower() if market.question else ""

    # ── Price features ──
    mid = market.midpoint or 0.5
    features["midpoint"] = mid
    features["price_distance_from_50"] = abs(mid - 0.5)
    features["is_longshot"] = 1.0 if mid < 0.15 or mid > 0.85 else 0.0
    features["is_near_even"] = 1.0 if 0.40 < mid < 0.60 else 0.0

    spread = market.spread or 0.04  # default if unknown
    features["spread"] = spread
    features["spread_pct"] = spread / mid if mid > 0 else 0.0

    # ── Volume / liquidity features ──
    try:
        vol = float(market.volume)
    except (ValueError, TypeError):
        vol = 0.0
    try:
        liq = float(market.liquidity)
    except (ValueError, TypeError):
        liq = 0.0

    features["log_volume"] = math.log1p(vol)
    features["log_liquidity"] = math.log1p(liq)
    features["volume_per_liquidity"] = vol / liq if liq > 0 else 0.0

    # Volume tiers
    features["vol_tier_micro"] = 1.0 if vol < 10_000 else 0.0
    features["vol_tier_small"] = 1.0 if 10_000 <= vol < 100_000 else 0.0
    features["vol_tier_medium"] = 1.0 if 100_000 <= vol < 1_000_000 else 0.0
    features["vol_tier_large"] = 1.0 if vol >= 1_000_000 else 0.0

    # ── Time features ──
    days_left = 30.0  # default
    if market.end_date:
        try:
            end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
            days_left = max(0, (end - datetime.now(timezone.utc)).total_seconds() / 86400)
        except (ValueError, TypeError):
            pass
    features["days_to_close"] = days_left
    features["log_days_to_close"] = math.log1p(days_left)
    features["is_expiring_soon"] = 1.0 if days_left < 3 else 0.0
    features["is_short_term"] = 1.0 if days_left < 7 else 0.0
    features["is_medium_term"] = 1.0 if 7 <= days_left < 30 else 0.0
    features["is_long_term"] = 1.0 if days_left >= 30 else 0.0

    age_days = 30.0
    if market.created_at:
        try:
            created = datetime.fromisoformat(market.created_at.replace("Z", "+00:00"))
            age_days = max(0, (datetime.now(timezone.utc) - created).total_seconds() / 86400)
        except (ValueError, TypeError):
            pass
    features["age_days"] = age_days
    features["log_age_days"] = math.log1p(age_days)
    features["is_new_market"] = 1.0 if age_days < 3 else 0.0

    # ── Text features ──
    features["question_length"] = float(len(market.question)) if market.question else 0.0
    features["question_word_count"] = float(len(q.split()))
    features["has_description"] = 1.0 if market.description and len(market.description) > 50 else 0.0
    features["description_length"] = float(len(market.description)) if market.description else 0.0
    features["has_number_in_question"] = 1.0 if re.search(r"\d", q) else 0.0
    features["has_question_mark"] = 1.0 if "?" in q else 0.0
    features["is_binary"] = 1.0 if len(market.outcomes) == 2 else 0.0
    features["num_outcomes"] = float(len(market.outcomes))

    # ── Category one-hot ──
    detected_category = "other"
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            detected_category = cat
            break
    for cat in CATEGORY_KEYWORDS:
        features[f"cat_{cat}"] = 1.0 if cat == detected_category else 0.0
    features["cat_other"] = 1.0 if detected_category == "other" else 0.0

    # ── Order book features ──
    if market.order_book:
        bids = market.order_book.get("bids", [])
        asks = market.order_book.get("asks", [])
        bid_depth = sum(float(b.get("size", 0)) for b in bids)
        ask_depth = sum(float(a.get("size", 0)) for a in asks)
        total_depth = bid_depth + ask_depth
        features["log_book_depth"] = math.log1p(total_depth)
        features["book_imbalance"] = abs(bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        features["has_thin_book"] = 1.0 if total_depth < 5_000 else 0.0
    else:
        features["log_book_depth"] = 0.0
        features["book_imbalance"] = 0.0
        features["has_thin_book"] = 1.0

    # ── Price history features ──
    if market.price_history:
        prices = [float(p.get("p", 0)) for p in market.price_history if p.get("p")]
        if len(prices) >= 2:
            features["price_range_7d"] = max(prices) - min(prices)
            features["price_trend_7d"] = prices[-1] - prices[0]
            mean_p = sum(prices) / len(prices)
            features["price_volatility_7d"] = (sum((p - mean_p) ** 2 for p in prices) / len(prices)) ** 0.5
        else:
            features["price_range_7d"] = 0.0
            features["price_trend_7d"] = 0.0
            features["price_volatility_7d"] = 0.0
    else:
        features["price_range_7d"] = 0.0
        features["price_trend_7d"] = 0.0
        features["price_volatility_7d"] = 0.0

    return features


# Ordered list of feature names for model input
FEATURE_NAMES: list[str] = sorted(extract_features(Market(
    id="dummy", question="test", description="", outcomes=["Yes", "No"],
    token_ids=["t"], end_date=None, active=True,
)).keys())


class HeuristicScorer:
    """Rule-based scorer as fallback when no trained model exists.

    Uses the same features as the ML model but with hand-tuned weights.
    Score > 0 means "worth analyzing with LLM".
    """

    def score(self, features: dict[str, float]) -> float:
        s = 0.0

        # Prefer markets near 50% — more room for LLM edge
        s += features.get("is_near_even", 0) * 1.5

        # Avoid extreme longshots — hard for LLMs too
        s -= features.get("is_longshot", 0) * 1.0

        # Prefer short-to-medium term — resolution validates predictions
        s += features.get("is_short_term", 0) * 1.5
        s += features.get("is_medium_term", 0) * 1.0
        s -= features.get("is_long_term", 0) * 0.5

        # Moderate volume — not too thin, not too efficient
        s += features.get("vol_tier_small", 0) * 1.5
        s += features.get("vol_tier_medium", 0) * 1.0
        s -= features.get("vol_tier_large", 0) * 0.5  # Very efficient market
        s -= features.get("vol_tier_micro", 0) * 0.5   # Too thin

        # Tight spread = better execution
        spread = features.get("spread", 0.04)
        if spread < 0.03:
            s += 1.0
        elif spread > 0.06:
            s -= 1.0

        # New markets = info asymmetry
        s += features.get("is_new_market", 0) * 1.0

        # Thin books = less competition
        s += features.get("has_thin_book", 0) * 0.5

        # Price movement = something is happening
        s += min(features.get("price_range_7d", 0) * 5.0, 1.5)

        # Good description = more context for LLM
        s += features.get("has_description", 0) * 0.5

        # Normalize to [0, 1]
        return 1.0 / (1.0 + math.exp(-s * 0.5))


class MarketPreScreener:
    """Pre-screens markets using ML features before expensive LLM analysis.

    Uses a trained scikit-learn model if available, otherwise falls back
    to a hand-tuned heuristic scorer.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.ML_PRESCREENER_MODEL_PATH
        self.model = None
        self.heuristic = HeuristicScorer()
        self._load_model()

    def _load_model(self):
        """Load trained model from disk if it exists."""
        path = Path(self.model_path)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded pre-screener model from %s", path)
            except Exception as e:
                logger.warning("Failed to load pre-screener model: %s", e)
                self.model = None

    def score(self, market: Market) -> float:
        """Return 0-1 score. Higher = more likely to have LLM edge."""
        features = extract_features(market)

        if self.model is not None:
            try:
                feature_vec = [features.get(name, 0.0) for name in FEATURE_NAMES]
                proba = self.model.predict_proba([feature_vec])[0]
                # Return probability of positive class (worth analyzing)
                return float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception as e:
                logger.warning("Model prediction failed, using heuristic: %s", e)

        return self.heuristic.score(features)

    def filter(self, markets: list[Market], threshold: float | None = None) -> list[Market]:
        """Keep only markets likely to benefit from LLM analysis.

        Returns markets sorted by score (best first).
        """
        threshold = threshold if threshold is not None else config.ML_PRESCREENER_THRESHOLD

        scored = [(m, self.score(m)) for m in markets]
        passed = [(m, s) for m, s in scored if s >= threshold]
        passed.sort(key=lambda x: x[1], reverse=True)

        total = len(markets)
        kept = len(passed)
        filtered_pct = (1 - kept / total) * 100 if total > 0 else 0
        logger.info("Pre-screener: %d/%d markets passed (%.0f%% filtered out, threshold=%.2f)",
                    kept, total, filtered_pct, threshold)

        return [m for m, s in passed]

    @staticmethod
    def train_from_history(db_module=None):
        """Train the pre-screener model from historical bet data.

        Requires resolved bets in the database. Call after accumulating
        enough data (50+ resolved bets recommended).

        Returns the trained model or None if insufficient data.
        """
        if db_module is None:
            from src import db as db_module

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.error("scikit-learn required for training. Install with: pip install scikit-learn")
            return None

        # Collect training data from resolved bets
        all_bets = []
        for trader_id in ["grok", "claude", "gemini"]:
            try:
                bets = db_module.get_resolved_bets(trader_id)
                all_bets.extend(bets)
            except Exception:
                continue

        if len(all_bets) < 30:
            logger.warning("Need at least 30 resolved bets for training, have %d", len(all_bets))
            return None

        # Build training set: need to reconstruct Market objects from bet data
        # This is approximate — we use what we can from stored bet data
        X = []
        y = []
        for bet in all_bets:
            # Create a minimal Market from bet data for feature extraction
            m = Market(
                id=bet.market_id,
                question=bet.market_question,
                description="",
                outcomes=["Yes", "No"],
                token_ids=[bet.token_id],
                end_date=None,
                active=False,
                category=bet.category,
            )
            m.midpoint = bet.entry_price
            features = extract_features(m)
            feature_vec = [features.get(name, 0.0) for name in FEATURE_NAMES]
            X.append(feature_vec)
            # Positive = bet was profitable
            y.append(1 if bet.pnl > 0 else 0)

        # Train
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X, y)

        # Cross-validate
        try:
            scores = cross_val_score(model, X, y, cv=min(5, len(X) // 10 or 2), scoring="accuracy")
            logger.info("Pre-screener CV accuracy: %.1f%% (+/- %.1f%%)",
                        scores.mean() * 100, scores.std() * 200)
        except Exception:
            pass

        # Save model
        model_path = Path(config.ML_PRESCREENER_MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved pre-screener model to %s (%d samples)", model_path, len(X))

        return model
