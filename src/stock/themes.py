"""Macro theme definitions for thematic stock investing.

Five themes with curated tickers. Theme weights come from config.
Two-layer approach: macro conviction (what to buy) x quant signals (when/how much).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import config


@dataclass
class ThemeTicker:
    """A ticker within a macro theme."""
    symbol: str
    conviction: float  # 0.0 - 1.0, how strongly this ticker represents the theme
    rationale: str


@dataclass
class MacroTheme:
    """A macro investment theme with associated tickers and sectors."""
    name: str
    weight: float  # from config, determines portfolio allocation bias
    sectors: list[str]  # GICS sectors
    tickers: list[ThemeTicker] = field(default_factory=list)
    description: str = ""


# --- Theme Definitions ---

def _build_themes() -> list[MacroTheme]:
    """Build all macro themes with current config weights."""
    return [
        MacroTheme(
            name="peak_oil",
            weight=config.STOCK_THEME_PEAK_OIL,
            sectors=["Energy"],
            description="Supply constraints drive energy prices higher",
            tickers=[
                ThemeTicker("XOM", 0.9, "Largest integrated oil major, strong dividends"),
                ThemeTicker("CVX", 0.9, "Second largest US oil major, Permian Basin exposure"),
                ThemeTicker("OXY", 0.8, "Buffett-backed, pure-play US shale + carbon capture"),
                ThemeTicker("HAL", 0.7, "Oilfield services, benefits from increased drilling"),
                ThemeTicker("DVN", 0.7, "Pure-play US shale, variable dividend model"),
            ],
        ),
        MacroTheme(
            name="china_rise",
            weight=config.STOCK_THEME_CHINA_RISE,
            sectors=["Consumer Discretionary", "Communication Services", "Industrials"],
            description="Economic rebalancing, US relative decline, China growth",
            tickers=[
                ThemeTicker("BABA", 0.9, "China e-commerce + cloud giant, deep value"),
                ThemeTicker("PDD", 0.8, "Pinduoduo + Temu, fast-growing e-commerce"),
                ThemeTicker("BIDU", 0.7, "China AI + search leader"),
                ThemeTicker("NIO", 0.6, "China EV leader, battery swap technology"),
                ThemeTicker("FXI", 0.5, "iShares China Large-Cap ETF, broad exposure"),
            ],
        ),
        MacroTheme(
            name="ai_blackswan",
            weight=config.STOCK_THEME_AI_BLACKSWAN,
            sectors=["Information Technology", "Communication Services"],
            description="Transformative AI winner-take-most dynamics",
            tickers=[
                ThemeTicker("NVDA", 0.95, "AI compute monopoly, data center GPUs"),
                ThemeTicker("MSFT", 0.8, "OpenAI partnership, Azure AI, enterprise AI adoption"),
                ThemeTicker("GOOGL", 0.8, "DeepMind, Gemini, AI-first search transformation"),
                ThemeTicker("TSM", 0.85, "TSMC fabricates all AI chips, irreplaceable"),
                ThemeTicker("AVGO", 0.7, "Custom AI accelerators (TPUs, etc.), networking"),
                ThemeTicker("AMD", 0.7, "AI GPU challenger, MI300X, data center growth"),
            ],
        ),
        MacroTheme(
            name="new_energy",
            weight=config.STOCK_THEME_NEW_ENERGY,
            sectors=["Utilities", "Energy", "Industrials"],
            description="Nuclear renaissance, grid modernization, clean energy",
            tickers=[
                ThemeTicker("NEE", 0.7, "NextEra Energy, largest wind/solar utility"),
                ThemeTicker("FSLR", 0.7, "First Solar, US-made solar panels, tariff beneficiary"),
                ThemeTicker("ENPH", 0.6, "Enphase microinverters, residential solar"),
                ThemeTicker("SMR", 0.8, "NuScale Power, small modular reactors"),
                ThemeTicker("CCJ", 0.85, "Cameco, world's largest uranium producer"),
                ThemeTicker("UUUU", 0.7, "Energy Fuels, US uranium + rare earths"),
                ThemeTicker("OKLO", 0.6, "Oklo, advanced fission microreactors"),
            ],
        ),
        MacroTheme(
            name="materials",
            weight=config.STOCK_THEME_MATERIALS,
            sectors=["Materials", "Industrials"],
            description="Copper, lithium, rare earths for energy transition",
            tickers=[
                ThemeTicker("FCX", 0.9, "Freeport-McMoRan, world's largest public copper miner"),
                ThemeTicker("ALB", 0.8, "Albemarle, largest lithium producer"),
                ThemeTicker("MP", 0.7, "MP Materials, only US rare earth mine"),
                ThemeTicker("NEM", 0.6, "Newmont, largest gold miner, inflation hedge"),
                ThemeTicker("VALE", 0.7, "Vale, iron ore + nickel, Brazil-based"),
            ],
        ),
    ]


def get_themes() -> list[MacroTheme]:
    """Return all macro themes with current config weights."""
    return _build_themes()


def get_all_theme_tickers() -> list[str]:
    """Return deduplicated list of all theme ticker symbols."""
    seen: set[str] = set()
    result: list[str] = []
    for theme in _build_themes():
        for ticker in theme.tickers:
            if ticker.symbol not in seen:
                seen.add(ticker.symbol)
                result.append(ticker.symbol)
    return result


def get_theme_score(symbol: str) -> dict[str, float]:
    """Return per-theme scores for a symbol.

    Returns dict mapping theme_name -> theme_weight * ticker_conviction.
    Empty dict if symbol is not in any theme.
    """
    scores: dict[str, float] = {}
    for theme in _build_themes():
        for ticker in theme.tickers:
            if ticker.symbol == symbol:
                scores[theme.name] = theme.weight * ticker.conviction
    return scores


def compute_composite_theme_score(symbol: str) -> float:
    """Compute composite theme score for a symbol.

    Sum of (theme_weight * ticker_conviction) across all themes.
    Returns 0.0 for non-theme stocks.
    """
    scores = get_theme_score(symbol)
    return sum(scores.values())


def get_sectors_for_symbol(symbol: str) -> list[str]:
    """Return GICS sectors associated with a symbol's themes."""
    sectors: set[str] = set()
    for theme in _build_themes():
        for ticker in theme.tickers:
            if ticker.symbol == symbol:
                sectors.update(theme.sectors)
    return sorted(sectors)


def get_theme_tickers_by_theme(theme_name: str) -> list[ThemeTicker]:
    """Return tickers for a specific theme."""
    for theme in _build_themes():
        if theme.name == theme_name:
            return theme.tickers
    return []
