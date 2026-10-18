"""Universe resolution: ETF holdings (live with static fallback), sectors, custom tickers."""

from __future__ import annotations

from typing import Callable

from finance_dashboard.models import UserInputError, dedupe_tickers, normalize_ticker
from finance_dashboard.recommender.models import RecommendationRequest

# Static fallbacks: live holdings via yfinance are preferred, but frequently
# return partial or empty data, so every supported ETF needs an offline list.
# Keys cover the major market-cap indices (SPY/QQQ/DIA/IWM/VTI/VOO) and the
# full set of Select Sector SPDR ETFs so a user can pick any slice they want.
# --- Full S&P 500 constituent list (circa 2024-2026 snapshot) ---
# This is the FULL set, not a top-N slice — so when yfinance's live holdings
# call lags or returns a partial result, the Recommender still has the whole
# S&P 500 to cluster across. SPY, VOO (both S&P 500 trackers), and VTI
# (total market, dominated by the same large-caps) all point at this list.
_SP500: list[str] = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APO",
    "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB",
    "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX",
    "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD",
    "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB",
    "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTLT",
    "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD",
    "DE", "DECK", "DELL", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR",
    "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
    "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV",
    "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES",
    "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO",
    "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
    "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV",
    "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI",
    "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM",
    "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS",
    "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX",
    "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK",
    "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH",
    "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE",
    "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS",
    "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE",
    "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC",
    "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
    "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL",
    "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX",
    "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG",
    "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS",
    "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER",
    "TFC", "TFX", "TGT", "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR", "TRGP",
    "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT",
    "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI",
    "USB", "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST",
    "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL",
    "WFC", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL",
    "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS",
]

# Full Nasdaq 100 constituent list (circa 2024-2026 snapshot).
_NASDAQ100: list[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CCEP",
    "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP",
    "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC",
    "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP",
    "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX",
    "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
    "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS",
    "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]

ETF_FALLBACK_HOLDINGS: dict[str, list[str]] = {
    # --- Broad-market / major indices ---
    # Point SPY / VOO / VTI at the full S&P 500 list. Live yfinance holdings
    # will override when available; this is the floor so a user who picks
    # "S&P 500" never gets a universe of 10 stocks.
    "SPY":  _SP500,
    "VOO":  _SP500,
    "VTI":  _SP500,
    "QQQ":  _NASDAQ100,
    "DIA":  ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
             "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
             "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"],
    # Russell 2000 is 2000 small-caps — ~40 representative picks here. Live
    # yfinance fetch returns the real top holdings; this is just a fallback.
    "IWM":  ["SMCI", "STRL", "ANF", "FTAI", "APPF", "ONTO", "MLI", "FLR", "FIX", "RMBS",
             "COMM", "BMI", "MARA", "RIOT", "APP", "SPSC", "CRS", "AIT", "CRVL", "ENSG",
             "ACIW", "AX", "BOX", "CARS", "CECO", "CENTA", "CHX", "CORT", "CRCT", "CTS",
             "DNOW", "ESE", "FBP", "FSS", "GPI", "HNI", "KRYS", "LANC", "MATX", "MLKN"],
    # --- SPDR Select Sector ETFs — full memberships (circa 2024-2026). ---
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "AMD", "ADBE", "ACN",
             "IBM", "INTU", "NOW", "TXN", "QCOM", "ADI", "AMAT", "LRCX", "KLAC", "MU",
             "PANW", "SNPS", "CDNS", "CRWD", "FTNT", "ANET", "APH", "PLTR", "ROP", "FICO",
             "MSI", "GLW", "HPQ", "HPE", "WDC", "STX", "KEYS", "ANSS", "SMCI", "ON"],
    "XLV":  ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "ISRG", "DHR",
             "PFE", "VRTX", "SYK", "BSX", "GILD", "ELV", "REGN", "MDT", "CI", "CVS",
             "HCA", "MCK", "BMY", "DXCM", "EW", "COR", "A", "IDXX", "GEHC", "IQV",
             "ZTS", "BIIB", "HUM", "BAX", "LH", "RMD", "STE", "WST", "WAT", "MRNA"],
    "XLF":  ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
             "BLK", "C", "SCHW", "CB", "PGR", "KKR", "MMC", "ICE", "CME", "AON",
             "USB", "PNC", "AIG", "COF", "TFC", "BX", "APO", "ALL", "AJG", "MET",
             "PRU", "TRV", "AFL", "BK", "MTB", "FITB", "NTRS", "RJF", "HIG", "DFS"],
    "XLE":  ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OXY", "WMB", "VLO",
             "HES", "KMI", "FANG", "DVN", "APA", "HAL", "BKR", "OKE", "TRGP", "CTRA"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "BKNG", "NKE", "LOW", "SBUX", "TJX", "ABNB",
             "ORLY", "HLT", "MAR", "GM", "F", "AZO", "ROST", "YUM", "DRI", "LULU",
             "LVS", "ULTA", "DECK", "DPZ", "RCL", "CCL", "EBAY", "KMX", "PHM", "DHI"],
    "XLP":  ["WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT",
             "KDP", "KVUE", "KMB", "GIS", "STZ", "SYY", "HSY", "KR", "KHC", "CHD"],
    "XLI":  ["GE", "CAT", "RTX", "HON", "UNP", "BA", "DE", "UPS", "LMT", "ETN",
             "ADP", "TT", "WM", "GEV", "NOC", "ITW", "EMR", "GD", "FDX", "CSX",
             "NSC", "PH", "JCI", "CARR", "URI", "PCAR", "CMI", "ROK", "FAST", "OTIS"],
    "XLU":  ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PCG",
             "PEG", "WEC", "ETR", "AWK", "DTE", "EIX", "AEE", "PPL", "ES", "CMS",
             "FE", "ATO", "CNP", "VST", "NI", "LNT", "EVRG", "PNW", "NRG", "AES"],
    "XLC":  ["META", "GOOGL", "GOOG", "NFLX", "DIS", "VZ", "T", "CMCSA", "EA", "TTWO",
             "TMUS", "CHTR", "OMC", "WBD", "MTCH", "LYV", "PARA", "NWSA", "FOXA", "IPG"],
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "O", "PSA", "CCI", "EXR",
             "AVB", "VICI", "SBAC", "CBRE", "WY", "EQR", "ESS", "ARE", "MAA", "INVH",
             "VTR", "UDR", "DOC", "IRM", "KIM", "CPT", "REG", "FRT", "HST", "BXP"],
    "XLB":  ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "CTVA", "DD", "NUE", "DOW",
             "VMC", "MLM", "PPG", "CF", "IFF", "STLD", "ALB", "EMN", "MOS", "BALL",
             "AMCR", "PKG", "LYB", "AVY", "FMC", "IP", "SW", "AXTA", "WRK", "TAP"],
    # --- Vanguard industry ETFs — same universes as the matching SPDRs. ---
    "VGT":  ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "CSCO", "AMD", "ADBE", "ACN",
             "IBM", "INTU", "NOW", "TXN", "QCOM", "ADI", "AMAT", "LRCX", "KLAC", "MU",
             "PANW", "SNPS", "CDNS", "CRWD", "FTNT", "ANET", "APH", "PLTR", "ROP", "FICO"],
    "VHT":  ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "ISRG", "DHR",
             "PFE", "VRTX", "SYK", "BSX", "GILD", "ELV", "REGN", "MDT", "CI", "CVS",
             "HCA", "MCK", "BMY", "DXCM", "EW", "COR", "A", "IDXX", "GEHC", "IQV"],
    "VFH":  ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
             "BLK", "C", "SCHW", "CB", "PGR", "KKR", "MMC", "ICE", "CME", "AON"],
}

# Full-name display label for each supported ETF. The UI pairs these with
# the symbol so a user who doesn't memorize three-letter ETF tickers can
# still tell which fund they're picking.
ETF_DISPLAY_NAMES: dict[str, str] = {
    "SPY":  "SPDR S&P 500 ETF",
    "VOO":  "Vanguard S&P 500 ETF",
    "QQQ":  "Invesco QQQ Trust (Nasdaq 100)",
    "DIA":  "SPDR Dow Jones Industrial Average",
    "IWM":  "iShares Russell 2000 ETF",
    "VTI":  "Vanguard Total Stock Market ETF",
    "XLK":  "Technology Select Sector SPDR",
    "XLV":  "Health Care Select Sector SPDR",
    "XLF":  "Financial Select Sector SPDR",
    "XLE":  "Energy Select Sector SPDR",
    "XLY":  "Consumer Discretionary Select Sector SPDR",
    "XLP":  "Consumer Staples Select Sector SPDR",
    "XLI":  "Industrial Select Sector SPDR",
    "XLU":  "Utilities Select Sector SPDR",
    "XLC":  "Communication Services Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
    "XLB":  "Materials Select Sector SPDR",
    "VGT":  "Vanguard Information Technology ETF",
    "VHT":  "Vanguard Health Care ETF",
    "VFH":  "Vanguard Financials ETF",
}

# Sector universes drawn from across the supported ETFs; selectable without an ETF.
PREDEFINED_UNIVERSES: dict[str, list[str]] = {
    "Tech":        ["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "AMD", "CRM", "ADBE", "ORCL"],
    "Medical":     ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "PFE", "AMGN", "ISRG", "VRTX"],
    "Finance":     ["JPM", "BRK-B", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "AXP"],
    "Energy":      ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OXY", "WMB", "VLO"],
    "Consumer":    ["AMZN", "TSLA", "HD", "MCD", "WMT", "PG", "COST", "KO", "PEP", "NKE"],
    "Industrials": ["CAT", "GE", "HON", "UNP", "BA", "DE", "UPS", "LMT", "ETN", "RTX"],
    "Communications": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "VZ", "T", "CMCSA", "EA", "TTWO"],
    "Utilities":   ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PCG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "O", "PSA", "CCI", "EXR"],
    "Materials":   ["LIN", "SHW", "ECL", "APD", "FCX", "NEM", "CTVA", "DD", "NUE", "DOW"],
}


def _yfinance_holdings(etf: str) -> list[str]:
    """Fetch top holdings for an ETF from Yahoo Finance."""
    import yfinance as yf

    holdings = yf.Ticker(etf).funds_data.top_holdings
    return [str(s) for s in holdings.index] if holdings is not None else []


class UniverseProvider:
    """Resolves a request's universe sources into one deduplicated ticker list."""

    def __init__(self, holdings_fetcher: Callable[[str], list[str]] | None = None):
        self._holdings_fetcher = holdings_fetcher or _yfinance_holdings

    def etf_holdings(self, etf: str) -> list[str]:
        if etf not in ETF_FALLBACK_HOLDINGS:
            raise UserInputError(
                f"Unknown ETF: {etf}. Supported: {', '.join(sorted(ETF_FALLBACK_HOLDINGS))}"
            )
        try:
            live = self._holdings_fetcher(etf)
        except Exception:
            live = []
        return [normalize_ticker(t) for t in live] or ETF_FALLBACK_HOLDINGS[etf]

    def resolve(self, request: RecommendationRequest) -> list[str]:
        """Union of ETF holdings, selected universes, and custom tickers, in that order."""
        tickers: list[str] = []
        if request.etf:
            tickers.extend(self.etf_holdings(request.etf))
        for universe in request.universes:
            if universe not in PREDEFINED_UNIVERSES:
                raise UserInputError(
                    f"Unknown universe: {universe}. "
                    f"Supported: {', '.join(sorted(PREDEFINED_UNIVERSES))}"
                )
            tickers.extend(PREDEFINED_UNIVERSES[universe])
        tickers.extend(normalize_ticker(t) for t in request.custom_tickers)

        tickers = dedupe_tickers([t for t in tickers if t])
        if not tickers:
            raise UserInputError(
                "Select at least one source: an ETF, a predefined universe, or custom tickers."
            )
        return tickers
