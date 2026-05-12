"""Deribit public data client with lightweight on-disk caching.

The project only uses public market-data endpoints, so no API key is needed.
Deribit option prices are quoted in the underlying coin for BTC/ETH options;
conversion into USD happens in ``preprocessing.py``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:  # requests is nicer when installed, but urllib keeps the module portable.
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    requests = None


DERIBIT_PROD = "https://www.deribit.com/api/v2"


class DeribitAPIError(RuntimeError):
    """Raised when Deribit returns an error payload or invalid response."""


@dataclass(frozen=True)
class DeribitClient:
    """Small public Deribit HTTP client.

    Parameters
    ----------
    base_url:
        API base URL. Production is used by default.
    cache_dir:
        Directory for cached JSON responses. Use ``None`` to disable caching.
    ttl_seconds:
        Cache time-to-live. Live option snapshots should be refreshed often,
        while repeated notebook runs should not hammer the public API.
    timeout:
        HTTP timeout in seconds.
    """

    base_url: str = DERIBIT_PROD
    cache_dir: Path | str | None = Path("data/cache")
    ttl_seconds: int = 300
    timeout: float = 20.0

    def _cache_path(self, method: str, params: Mapping[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        encoded = json.dumps({"method": method, "params": dict(params)}, sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        safe_method = method.replace("/", "_")
        return Path(self.cache_dir) / f"{safe_method}_{digest}.json"

    def get_json(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Call a public Deribit method and return the decoded JSON payload."""

        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_path = self._cache_path(method, params)

        if use_cache and cache_path and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age <= self.ttl_seconds:
                return json.loads(cache_path.read_text())

        url = f"{self.base_url.rstrip('/')}/{method}"
        if requests is not None:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        else:  # pragma: no cover
            query = urllib.parse.urlencode(params)
            request_url = f"{url}?{query}" if query else url
            try:
                with urllib.request.urlopen(request_url, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise DeribitAPIError(f"Deribit request failed: {exc}") from exc

        if "error" in payload:
            raise DeribitAPIError(str(payload["error"]))
        if "result" not in payload:
            raise DeribitAPIError(f"Unexpected Deribit response: {payload}")

        if use_cache and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        return payload

    def get_option_book(self, currency: str, *, use_cache: bool = True) -> pd.DataFrame:
        """Fetch current option book summary for BTC or ETH."""

        payload = self.get_json(
            "public/get_book_summary_by_currency",
            {"currency": currency.upper(), "kind": "option"},
            use_cache=use_cache,
        )
        return pd.DataFrame(payload["result"])

    def get_instruments(self, currency: str, *, use_cache: bool = True) -> pd.DataFrame:
        """Fetch active option instrument metadata for BTC or ETH."""

        payload = self.get_json(
            "public/get_instruments",
            {"currency": currency.upper(), "kind": "option", "expired": "false"},
            use_cache=use_cache,
        )
        return pd.DataFrame(payload["result"])

    def get_index_history(
        self,
        currency: str,
        *,
        range_: str = "1y",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch Deribit index price history for BTC or ETH."""

        index_name = f"{currency.lower()}_usd"
        payload = self.get_json(
            "public/get_index_chart_data",
            {"index_name": index_name, "range": range_},
            use_cache=use_cache,
        )
        frame = pd.DataFrame(payload["result"], columns=["timestamp", "price"])
        frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame["currency"] = currency.upper()
        return frame[["currency", "timestamp", "datetime", "price"]]


def fetch_market_snapshot(
    currencies: tuple[str, ...] = ("BTC", "ETH"),
    *,
    client: DeribitClient | None = None,
    use_cache: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch option books, instrument metadata, and index histories."""

    client = client or DeribitClient()
    snapshot: dict[str, dict[str, pd.DataFrame]] = {}
    for currency in currencies:
        symbol = currency.upper()
        snapshot[symbol] = {
            "book": client.get_option_book(symbol, use_cache=use_cache),
            "instruments": client.get_instruments(symbol, use_cache=use_cache),
            "index": client.get_index_history(symbol, use_cache=use_cache),
        }
    return snapshot

