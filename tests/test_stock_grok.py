"""Tests for the stock_grok LLM agent — hybrid quant+Grok stock analysis."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import config
from src.models import Analysis, Market, Recommendation, STOCK_TRADER_IDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stock_market():
    """A sample stock Market."""
    return Market(
        id="stock_AAPL",
        question="Stock: AAPL",
        description="Stock position in AAPL",
        outcomes=["LONG", "SHORT"],
        token_ids=["AAPL", "AAPL"],
        end_date=None,
        active=True,
        volume="5000000",
        liquidity="1000000",
        midpoint=185.50,
        spread=0.01,
        ohlcv=[
            {"t": "2026-03-01", "o": 180.0, "h": 186.0, "l": 179.5, "c": 185.0, "v": 1000000},
            {"t": "2026-03-02", "o": 185.0, "h": 187.0, "l": 184.0, "c": 186.0, "v": 1100000},
            {"t": "2026-03-03", "o": 186.0, "h": 188.0, "l": 185.5, "c": 185.5, "v": 900000},
        ],
        market_system="stock",
        symbol="AAPL",
        sector="Technology",
        theme_scores={"ai_blackswan": 0.8, "new_energy": 0.1},
    )


@pytest.fixture
def quant_analysis():
    """A sample quant Analysis with signal context."""
    return Analysis(
        market_id="stock_AAPL",
        model="stock_quant",
        recommendation=Recommendation.BUY_YES,
        confidence=0.65,
        estimated_probability=0.62,
        reasoning="Stock: AAPL; Theme score: 0.80; momentum: bullish",
        category="Technology",
        extras={
            "agent": "stock_quant",
            "signals": [
                {"name": "momentum", "direction": "bullish", "strength": 0.7},
                {"name": "rsi", "direction": "bullish", "strength": 0.55},
            ],
            "theme_score": 0.80,
            "direction": "bullish",
        },
    )


def _mock_grok_response(recommendation="BUY_YES", confidence=0.72, est_prob=0.68,
                         reasoning="Strong momentum with positive theme alignment"):
    """Create a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({
        "recommendation": recommendation,
        "confidence": confidence,
        "estimated_probability": est_prob,
        "reasoning": reasoning,
    })
    response.usage = MagicMock()
    response.usage.prompt_tokens = 500
    response.usage.completion_tokens = 100
    return response


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestTraderRegistration:
    def test_stock_grok_in_trader_ids(self):
        assert "stock_grok" in STOCK_TRADER_IDS

    def test_stock_grok_config_defaults(self):
        assert hasattr(config, "STOCK_GROK_ENABLED")
        assert hasattr(config, "STOCK_GROK_MIN_CONFIDENCE")
        assert hasattr(config, "STOCK_GROK_TOP_N")
        assert config.STOCK_GROK_MIN_CONFIDENCE == 0.40
        assert config.STOCK_GROK_TOP_N == 10


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_builds_prompt_with_symbol(self, stock_market, quant_analysis):
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, quant_analysis)
        assert "AAPL" in prompt
        assert "185.50" in prompt

    def test_includes_sector(self, stock_market, quant_analysis):
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, quant_analysis)
        assert "Technology" in prompt

    def test_includes_ohlcv(self, stock_market, quant_analysis):
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, quant_analysis)
        assert "OHLCV" in prompt
        assert "180.0" in prompt  # first bar open

    def test_includes_quant_signals(self, stock_market, quant_analysis):
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, quant_analysis)
        assert "momentum" in prompt
        assert "bullish" in prompt
        assert "0.65" in prompt  # quant confidence

    def test_includes_theme_scores(self, stock_market, quant_analysis):
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, quant_analysis)
        assert "ai_blackswan" in prompt

    def test_works_without_quant(self, stock_market):
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, None)
        assert "AAPL" in prompt
        assert "Quant Signals" not in prompt

    def test_works_without_ohlcv(self, stock_market):
        stock_market.ohlcv = None
        from src.stock.analyzer import _build_stock_prompt
        prompt = _build_stock_prompt(stock_market, None)
        assert "AAPL" in prompt
        assert "OHLCV" not in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_parses_clean_json(self):
        from src.stock.analyzer import _parse_response
        text = '{"recommendation": "BUY_YES", "confidence": 0.8}'
        result = _parse_response(text, "test")
        assert result["recommendation"] == "BUY_YES"

    def test_parses_markdown_fenced_json(self):
        from src.stock.analyzer import _parse_response
        text = '```json\n{"recommendation": "SKIP", "confidence": 0.2}\n```'
        result = _parse_response(text, "test")
        assert result["recommendation"] == "SKIP"

    def test_parses_json_with_surrounding_text(self):
        from src.stock.analyzer import _parse_response
        text = 'Here is my analysis:\n{"recommendation": "BUY_YES", "confidence": 0.7}\nDone.'
        result = _parse_response(text, "test")
        assert result["recommendation"] == "BUY_YES"

    def test_returns_empty_on_invalid(self):
        from src.stock.analyzer import _parse_response
        result = _parse_response("not json at all", "test")
        assert result == {}


# ---------------------------------------------------------------------------
# Analyzer integration
# ---------------------------------------------------------------------------

class TestStockGrokAnalyzer:
    @patch("src.stock.analyzer.StockGrokAnalyzer.__init__", return_value=None)
    def test_analyze_buy_yes(self, mock_init, stock_market, quant_analysis):
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer.__new__(StockGrokAnalyzer)
        analyzer.model = "grok-test"
        analyzer.client = MagicMock()
        analyzer.client.chat.completions.create.return_value = _mock_grok_response()

        result = analyzer.analyze(stock_market, quant_analysis)

        assert result.model == "stock_grok"
        assert result.recommendation == Recommendation.BUY_YES
        assert result.confidence == 0.72
        assert result.estimated_probability == 0.68
        assert result.extras["agent"] == "stock_grok"
        assert result.extras["has_quant_context"] is True

    @patch("src.stock.analyzer.StockGrokAnalyzer.__init__", return_value=None)
    def test_analyze_skip(self, mock_init, stock_market):
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer.__new__(StockGrokAnalyzer)
        analyzer.model = "grok-test"
        analyzer.client = MagicMock()
        analyzer.client.chat.completions.create.return_value = _mock_grok_response(
            recommendation="SKIP", confidence=0.3, reasoning="Weak signals"
        )

        result = analyzer.analyze(stock_market)

        assert result.recommendation == Recommendation.SKIP
        assert result.extras["has_quant_context"] is False

    @patch("src.stock.analyzer.StockGrokAnalyzer.__init__", return_value=None)
    def test_analyze_bad_json_returns_skip(self, mock_init, stock_market):
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer.__new__(StockGrokAnalyzer)
        analyzer.model = "grok-test"
        analyzer.client = MagicMock()

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "I can't analyze this"
        response.usage = None
        analyzer.client.chat.completions.create.return_value = response

        result = analyzer.analyze(stock_market)
        assert result.recommendation == Recommendation.SKIP
        assert "Failed to parse" in result.reasoning

    @patch("src.stock.analyzer.StockGrokAnalyzer.__init__", return_value=None)
    def test_analyze_api_error_raises(self, mock_init, stock_market):
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer.__new__(StockGrokAnalyzer)
        analyzer.model = "grok-test"
        analyzer.client = MagicMock()
        analyzer.client.chat.completions.create.side_effect = TimeoutError("API timeout")

        with pytest.raises(TimeoutError):
            analyzer.analyze(stock_market)

    @patch("src.stock.analyzer.StockGrokAnalyzer.__init__", return_value=None)
    def test_confidence_clamped(self, mock_init, stock_market):
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer.__new__(StockGrokAnalyzer)
        analyzer.model = "grok-test"
        analyzer.client = MagicMock()
        analyzer.client.chat.completions.create.return_value = _mock_grok_response(
            confidence=1.5, est_prob=-0.2
        )

        result = analyzer.analyze(stock_market)
        assert result.confidence == 1.0
        assert result.estimated_probability == 0.0

    @patch("src.stock.analyzer.StockGrokAnalyzer.__init__", return_value=None)
    def test_quant_context_in_extras(self, mock_init, stock_market, quant_analysis):
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer.__new__(StockGrokAnalyzer)
        analyzer.model = "grok-test"
        analyzer.client = MagicMock()
        analyzer.client.chat.completions.create.return_value = _mock_grok_response()

        result = analyzer.analyze(stock_market, quant_analysis)
        assert result.extras["quant_direction"] == "bullish"
        assert result.extras["quant_confidence"] == 0.65
        assert len(result.extras["quant_signals"]) == 2


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

class TestRunnerIntegration:
    def test_run_stock_grok_no_bullish(self, stock_market):
        from src.stock.runner import _run_stock_grok, StockCycleResult
        result = StockCycleResult()

        # No bullish analyses → 0 trades
        bearish_analysis = Analysis(
            market_id="stock_X",
            model="stock_quant",
            recommendation=Recommendation.SKIP,
            confidence=0.0,
            estimated_probability=0.0,
            reasoning="bearish",
        )
        trades = _run_stock_grok(
            quant_analyses=[(stock_market, bearish_analysis)],
            all_markets=[stock_market],
            quant_sim=MagicMock(),
            api=None,
            result=result,
            _status=lambda msg, trader_id="stock_grok": None,
            cycle_number=1,
        )
        assert trades == 0

    @patch("src.stock.analyzer.StockGrokAnalyzer")
    def test_run_stock_grok_places_trade(self, MockAnalyzer, stock_market, quant_analysis):
        from src.stock.runner import _run_stock_grok, StockCycleResult

        # Mock analyzer
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.analyze.return_value = Analysis(
            market_id="stock_AAPL",
            model="stock_grok",
            recommendation=Recommendation.BUY_YES,
            confidence=0.72,
            estimated_probability=0.68,
            reasoning="Strong buy",
            extras={"agent": "stock_grok"},
        )

        # Mock simulator at source (imported inside function)
        mock_bet = MagicMock()
        mock_bet.amount = 50.0
        mock_sim = MagicMock()
        mock_sim.place_trade.return_value = mock_bet

        result = StockCycleResult()
        with patch("src.stock.simulator.StockSimulator", return_value=mock_sim):
            trades = _run_stock_grok(
                quant_analyses=[(stock_market, quant_analysis)],
                all_markets=[stock_market],
                quant_sim=MagicMock(),
                api=None,
                result=result,
                _status=lambda msg, trader_id="stock_grok": None,
                cycle_number=1,
            )
        assert trades == 1
        mock_analyzer.analyze.assert_called_once()

    @patch("src.stock.analyzer.StockGrokAnalyzer")
    def test_run_stock_grok_respects_confidence_gate(self, MockAnalyzer, stock_market, quant_analysis):
        from src.stock.runner import _run_stock_grok, StockCycleResult

        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.analyze.return_value = Analysis(
            market_id="stock_AAPL",
            model="stock_grok",
            recommendation=Recommendation.BUY_YES,
            confidence=0.20,  # Below STOCK_GROK_MIN_CONFIDENCE (0.40)
            estimated_probability=0.55,
            reasoning="Low confidence",
            extras={"agent": "stock_grok"},
        )

        result = StockCycleResult()
        with patch("src.stock.simulator.StockSimulator", return_value=MagicMock()):
            trades = _run_stock_grok(
                quant_analyses=[(stock_market, quant_analysis)],
                all_markets=[stock_market],
                quant_sim=MagicMock(),
                api=None,
                result=result,
                _status=lambda msg, trader_id="stock_grok": None,
                cycle_number=1,
            )
        assert trades == 0


# ---------------------------------------------------------------------------
# StockCycleResult
# ---------------------------------------------------------------------------

class TestStockCycleResult:
    def test_trades_by_trader_field(self):
        from src.stock.runner import StockCycleResult
        r = StockCycleResult()
        assert r.trades_by_trader == {}
        r.trades_by_trader["stock_grok"] = 3
        r.trades_by_trader["stock_quant"] = 5
        assert r.trades_by_trader["stock_grok"] == 3
