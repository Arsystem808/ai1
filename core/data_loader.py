from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os, pathlib, requests, pandas as pd

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str

class DataLoader:
    def __init__(self, polygon_api_key: str | None = None):
        self.polygon_api_key = polygon_api_key or os.getenv("POLYGON_API_KEY")

    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> FetchResult:
        # 1) Polygon
        if self.polygon_api_key:
            try:
                df = self._polygon_history(symbol, interval=interval, lookback_days=365)
                if df is not None and not df.empty:
                    return FetchResult(df=df, source="polygon")
            except Exception:
                pass
        # 2) Yahoo
        try:
            import yfinance as yf
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns=str.title)
                return FetchResult(df=df[["Date","Open","High","Low","Close","Volume"]], source="yahoo")
        except Exception:
            pass
        # 3) Demo CSV
        here = pathlib.Path(__file__).resolve().parent.parent
        demo = here / "data" / "demo" / f"{symbol.lower()}_demo.csv"
        if demo.exists():
            df = pd.read_csv(demo)
            return FetchResult(df=df[["Date","Open","High","Low","Close","Volume"]], source="demo-csv")
        raise RuntimeError("No data from Polygon, Yahoo, or demo CSV.")

    def _polygon_history(self, symbol: str, interval: str = "1d", lookback_days: int = 365) -> pd.DataFrame | None:
        if interval != "1d":
            return None
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self.polygon_api_key}"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        if not js or "results" not in js or not js["results"]:
            return None
        rows = []
        for it in js["results"]:
            rows.append({
                "Date": pd.to_datetime(it["t"], unit="ms").tz_localize("UTC").tz_convert(None),
                "Open": it["o"], "High": it["h"], "Low": it["l"], "Close": it["c"], "Volume": it.get("v", 0)
            })
        return pd.DataFrame(rows)
