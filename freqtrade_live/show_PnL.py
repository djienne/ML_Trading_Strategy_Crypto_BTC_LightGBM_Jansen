#!/usr/bin/env python3
"""
Show performance for LightGBM strategy Freqtrade containers.
- Uses docker ps to get container names + host port -> 8080
- Reads API credentials from freqtrade_live/user_data/config.json
- Binary good/bad colors + CAGR from profit_all% since first trade
- DAYS column = days since first trade
- LAST TRADE column = time since most recent open/close trade event
- SHARPE column = annualized Sharpe of per-bar returns since the first trade,
  with flat bars counted as 0 — the same convention as the vectorized backtest
  (src/backtest.py), so it is comparable to grid-search Sharpe numbers
"""

import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "user_data" / "config.json"

TIMEOUT = 3  # seconds
CONTAINER_KEYWORD = "lgbm"  # Filter containers containing this keyword

PORT_RE = re.compile(r"(?:\d{1,3}(?:\.\d{1,3}){3}:)?(\d+)->8080/tcp")
ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def load_api_auth() -> HTTPBasicAuth:
    """Read API credentials from the project config."""
    try:
        api = json.loads(CONFIG_PATH.read_text()).get("api_server", {})
        return HTTPBasicAuth(api.get("username", "freqtrader"), api.get("password", ""))
    except (FileNotFoundError, json.JSONDecodeError):
        return HTTPBasicAuth("freqtrader", "")


# ---- ANSI colors ----

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_WHITE = '\033[97m'


NO_DATA = f"{Colors.DIM}-{Colors.RESET}"


def good_bad(text: str, good: bool) -> str:
    color = Colors.BRIGHT_GREEN if good else Colors.BRIGHT_RED
    return f"{color}{text}{Colors.RESET}"


def colorize_profit(value: Optional[float]) -> str:
    return NO_DATA if value is None else good_bad(f"{value:.2f}%", value > 0)


def colorize_win_rate(value: Optional[float]) -> str:
    return NO_DATA if value is None else good_bad(f"{value:.2f}%", value >= 50)


def colorize_profit_factor(pf_str: str) -> str:
    if pf_str == "-":
        return NO_DATA
    try:
        return good_bad(pf_str, float(pf_str) >= 1.0)
    except ValueError:
        return f"{Colors.DIM}{pf_str}{Colors.RESET}"


def colorize_sharpe(value: Optional[float]) -> str:
    return NO_DATA if value is None else good_bad(f"{value:.2f}", value >= 1.0)


def colorize_cagr(value: Optional[float]) -> str:
    return NO_DATA if value is None else good_bad(f"{value:.2f}%", value > 10)


def colorize_drawdown(value: Optional[float]) -> str:
    if value is None:
        return NO_DATA
    color = Colors.WHITE if value == 0 else Colors.BRIGHT_RED
    return f"{color}{value:.2f}%{Colors.RESET}"


def colorize_trades(trades: Any) -> str:
    if trades == "-":
        return NO_DATA
    try:
        return f"{int(trades)}"
    except (ValueError, TypeError):
        return f"{Colors.DIM}{trades}{Colors.RESET}"


def colorize_last_trade(value: Optional[str], timestamp_ms: Optional[int]) -> str:
    if not value or timestamp_ms is None:
        return NO_DATA
    try:
        trade_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        elapsed_hours = (datetime.now(timezone.utc) - trade_date).total_seconds() / 3600.0
    except Exception:
        return f"{Colors.DIM}{value}{Colors.RESET}"
    if elapsed_hours <= 24:
        return f"{Colors.BRIGHT_GREEN}{value}{Colors.RESET}"
    if elapsed_hours <= 24 * 7:
        return f"{Colors.YELLOW}{value}{Colors.RESET}"
    return f"{Colors.BRIGHT_RED}{value}{Colors.RESET}"


def colorize_tagged(value: Any, color: str) -> str:
    return NO_DATA if value == "-" else f"{color}{value}{Colors.RESET}"


# ---- Robust date parsing helpers ----

def try_parse_dt(val: Any) -> Optional[int]:
    """
    Parse various datetime representations commonly returned by Freqtrade.
    Returns milliseconds since epoch (UTC) or None.
    Accepts ISO strings (with/without 'Z'), epoch seconds/ms (int/float), or dicts with common keys.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if val > 1e12:      # ms
            return int(val)
        if val > 1e9:       # s
            return int(val * 1000)
        return None
    if isinstance(val, str):
        s = val.strip()
        try:
            if s.isdigit():
                return try_parse_dt(int(s))
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    if isinstance(val, dict):
        for k in ("open_date", "open_at", "open_time", "opened_at", "date_open", "date"):
            ts = try_parse_dt(val.get(k))
            if ts:
                return ts
    return None


def extract_first_ts_from_any(obj: Any, keys: Iterable[str]) -> Optional[int]:
    """Search dict OR list-of-dicts for the first parseable timestamp in given keys."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in keys:
            ts = try_parse_dt(obj.get(k))
            if ts:
                return ts
        return None
    if isinstance(obj, list):
        for item in obj:
            ts = extract_first_ts_from_any(item, keys)
            if ts:
                return ts
        return None
    return None


def extract_earliest_open_ts_from_trades(trades_list: Iterable[Dict[str, Any]]) -> Optional[int]:
    candidates: List[int] = []
    for t in trades_list or []:
        for key in ("open_date", "open_at", "open_time", "opened_at", "date_open", "open_timestamp"):
            ts = try_parse_dt(t.get(key))
            if ts:
                candidates.append(ts)
                break
    return min(candidates) if candidates else None


def extract_latest_ts_from_trades(trades_list: Iterable[Dict[str, Any]]) -> Optional[int]:
    candidates: List[int] = []
    trade_keys = (
        "close_timestamp",
        "close_date",
        "open_fill_timestamp",
        "open_fill_date",
        "open_timestamp",
        "open_date",
        "date",
    )
    order_keys = (
        "order_filled_timestamp",
        "order_filled_date",
        "order_timestamp",
        "order_date",
    )
    for trade in trades_list or []:
        for key in trade_keys:
            ts = try_parse_dt(trade.get(key))
            if ts:
                candidates.append(ts)
        orders = trade.get("orders")
        if isinstance(orders, list):
            for order in orders:
                if not isinstance(order, dict):
                    continue
                for key in order_keys:
                    ts = try_parse_dt(order.get(key))
                    if ts:
                        candidates.append(ts)
    return max(candidates) if candidates else None


# ---- IO helpers ----

def docker_containers() -> List[Dict[str, Optional[int]]]:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"], text=True
        )
    except Exception:
        return []
    rows: List[Dict[str, Optional[int]]] = []
    for line in out.splitlines():
        name, *port_parts = line.split("\t", 1)
        ports = port_parts[0] if port_parts else ""
        m = PORT_RE.search(ports or "")
        port = int(m.group(1)) if m else None
        rows.append({"name": name.strip(), "port": port})
    return rows


def get_json(url: str, auth: HTTPBasicAuth) -> Optional[Any]:
    try:
        r = requests.get(url, auth=auth, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def normalize_trades(data: Any) -> List[Dict[str, Any]]:
    """Unwrap the various shapes /trades-style endpoints return into a list of dicts."""
    if isinstance(data, dict):
        data = data.get("trades") or data.get("data") or []
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict)]


def fetch_trades_list(base: str, auth: HTTPBasicAuth) -> List[Dict[str, Any]]:
    for url in (f"{base}/trades?limit=5000", f"{base}/trades"):
        trades = normalize_trades(get_json(url, auth))
        if trades:
            return trades
    return []


def get_first_trade_timestamp(
    base: str,
    auth: HTTPBasicAuth,
    prof: Dict[str, Any],
    trades_list: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Robustly find the timestamp (ms) of the earliest trade (open or closed).
    Try, in order:
      1) /status (dict OR list) fields like 'first_trade_date' or 'first_trade_timestamp'
      2) hints on /profit
      3) the prefetched /trades list → earliest open date across items
      4) /closed_trades then /open_trades (fallbacks)
    """
    status = get_json(f"{base}/status", auth)
    ts = extract_first_ts_from_any(status, ("first_trade_date", "first_trade_timestamp", "first_trade"))
    if ts:
        return ts

    ts = extract_first_ts_from_any(prof, ("first_trade_date", "first_trade_timestamp"))
    if ts:
        return ts

    ts = extract_earliest_open_ts_from_trades(trades_list)
    if ts:
        return ts

    for endpoint in ("closed_trades", "open_trades"):
        data = get_json(f"{base}/{endpoint}", auth)
        if isinstance(data, dict) and endpoint in data:
            data = data[endpoint]
        ts = extract_earliest_open_ts_from_trades(normalize_trades(data))
        if ts:
            return ts

    return None


def get_last_trade_timestamp(
    trades_list: List[Dict[str, Any]],
    prof: Dict[str, Any],
) -> Optional[int]:
    """
    Find the newest known trade timestamp in milliseconds.
    Prefer full trade/order details, then fall back to /profit's latest trade hint.
    """
    ts = extract_latest_ts_from_trades(trades_list)
    if ts:
        return ts
    return extract_first_ts_from_any(
        prof,
        ("latest_trade_timestamp", "latest_trade_date", "latest_trade"),
    )


# ---- Metrics ----

def calculate_cagr(profit_all_percent: Optional[float], first_trade_timestamp_ms: Optional[int]) -> Optional[float]:
    """
    CAGR from current PnL of ALL trades (profit_all_percent) and time since first trade.
    Annualizes even for short histories. Returns % or None if inputs invalid.
    """
    if profit_all_percent is None or first_trade_timestamp_ms is None:
        return None
    try:
        first_trade_date = datetime.fromtimestamp(first_trade_timestamp_ms / 1000, tz=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - first_trade_date).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return None

        years_elapsed = elapsed_days / 365.25
        ending_value = 100.0 + float(profit_all_percent)
        if ending_value <= 0:
            return None  # nuked account case

        return (pow(ending_value / 100.0, 1.0 / years_elapsed) - 1.0) * 100.0
    except Exception:
        return None


def timeframe_minutes(timeframe: Optional[str]) -> int:
    """Parse a freqtrade timeframe ('15m', '1h', '1d') to minutes; default 15."""
    if not timeframe:
        return 15
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    try:
        return int(timeframe[:-1]) * units[timeframe[-1]]
    except (KeyError, ValueError):
        return 15


def calculate_annualized_sharpe(
    trades_list: Iterable[Dict[str, Any]],
    timeframe: Optional[str],
) -> Optional[float]:
    """
    Annualized Sharpe of per-bar returns since the first trade, flat bars = 0.

    Each closed trade's profit ratio is spread geometrically over its held
    bars; every other bar between the first trade and now contributes 0. This
    matches the vectorized backtest convention in src/backtest.py
    (mean/std of per-bar net returns * sqrt(bars_per_year)).
    """
    tf_min = timeframe_minutes(timeframe)
    bar_ms = tf_min * 60_000

    bar_returns: Dict[int, float] = {}
    first_bar: Optional[int] = None
    for t in trades_list or []:
        profit = t.get("close_profit")
        open_ts = try_parse_dt(t.get("open_timestamp") or t.get("open_date"))
        close_ts = try_parse_dt(t.get("close_timestamp") or t.get("close_date"))
        if open_ts is None:
            continue
        open_bar = open_ts // bar_ms
        first_bar = open_bar if first_bar is None else min(first_bar, open_bar)
        if profit is None or close_ts is None:
            continue  # open trade: counts for the window start only
        held = max(1, close_ts // bar_ms - open_bar)
        per_bar = (1.0 + float(profit)) ** (1.0 / held) - 1.0
        for b in range(open_bar, open_bar + held):
            bar_returns[b] = bar_returns.get(b, 0.0) + per_bar

    if first_bar is None or not bar_returns:
        return None
    now_bar = int(datetime.now(timezone.utc).timestamp() * 1000) // bar_ms
    n = now_bar - first_bar + 1
    if n < 2:
        return None

    total = sum(bar_returns.values())
    total_sq = sum(r * r for r in bar_returns.values())
    mean = total / n
    var = (total_sq - n * mean * mean) / (n - 1)
    if var <= 0:
        return None
    bars_per_year = 365.25 * 24 * 60 / tf_min
    return mean / math.sqrt(var) * math.sqrt(bars_per_year)


def days_since_first_trade(first_trade_timestamp_ms: Optional[int]) -> Optional[int]:
    if first_trade_timestamp_ms is None:
        return None
    try:
        first_trade_date = datetime.fromtimestamp(first_trade_timestamp_ms / 1000, tz=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - first_trade_date).total_seconds() / 86400.0
        if elapsed_days < 0:
            return None
        return int(elapsed_days)  # floor to whole days
    except Exception:
        return None


def format_since_timestamp(timestamp_ms: Optional[int]) -> Optional[str]:
    if timestamp_ms is None:
        return None
    try:
        trade_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        elapsed_seconds = int((datetime.now(timezone.utc) - trade_date).total_seconds())
    except Exception:
        return None
    if elapsed_seconds < 0:
        return None
    if elapsed_seconds < 60:
        return "now"
    minutes = elapsed_seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        return f"{hours}h {mins}m ago" if mins else f"{hours}h ago"
    days = hours // 24
    hrs = hours % 24
    return f"{days}d {hrs}h ago" if hrs else f"{days}d ago"


# ---- Table rendering ----

HEADERS = [
    ("CONTAINER", "container"),
    ("PORT", "port"),
    ("BOT", "bot"),
    ("STRATEGY", "strategy"),
    ("TRADES", "trades"),
    ("LAST TRADE", "last_trade"),
    ("WIN RATE", "win_rate"),
    ("PROFIT ALL", "profit_all"),
    ("PROFIT CLOSED", "profit_closed"),
    ("PF", "pf"),
    ("SHARPE", "sharpe"),
    ("MAX DD", "max_dd"),
    ("DAYS", "days"),
    ("CAGR", "cagr"),
]

CELL_RENDERERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "container": lambda r: f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{r['container']}{Colors.RESET}",
    "port": lambda r: colorize_tagged(r["port"], Colors.BRIGHT_MAGENTA),
    "bot": lambda r: colorize_tagged(r["bot"], Colors.CYAN),
    "strategy": lambda r: colorize_tagged(r["strategy"], Colors.YELLOW),
    "trades": lambda r: colorize_trades(r["trades"]),
    "last_trade": lambda r: colorize_last_trade(r["last_trade"], r["last_trade_ts"]),
    "win_rate": lambda r: colorize_win_rate(r["win_rate"]),
    "profit_all": lambda r: colorize_profit(r["profit_all"]),
    "profit_closed": lambda r: colorize_profit(r["profit_closed"]),
    "pf": lambda r: colorize_profit_factor(str(r["pf"])),
    "sharpe": lambda r: colorize_sharpe(r["sharpe"]),
    "max_dd": lambda r: colorize_drawdown(r["max_dd"]),
    "days": lambda r: NO_DATA if r["days"] is None else str(r["days"]),
    "cagr": lambda r: colorize_cagr(r["cagr"]),
}


def make_row(name: str, **values: Any) -> Dict[str, Any]:
    """Row dict with no-data defaults; pass keyword overrides for known fields."""
    row: Dict[str, Any] = {
        "container": name,
        "port": "-",
        "bot": "-",
        "strategy": "-",
        "trades": "-",
        "last_trade": None,
        "last_trade_ts": None,
        "win_rate": None,
        "profit_all": None,
        "profit_closed": None,
        "pf": "-",
        "sharpe": None,
        "max_dd": None,
        "days": None,
        "cagr": None,
    }
    row.update(values)
    return row


def plain_text_len(text: str) -> int:
    return len(ANSI_RE.sub('', text))


def container_row(name: str, port: int, auth: HTTPBasicAuth) -> Dict[str, Any]:
    base = f"http://127.0.0.1:{port}/api/v1"
    cfg = get_json(f"{base}/show_config", auth) or {}
    bot_name = cfg.get("bot_name") or "-"
    strategy = cfg.get("strategy") or "-"

    prof = get_json(f"{base}/profit", auth)
    if not isinstance(prof, dict) or not prof:
        return make_row(name, port=port, bot=bot_name, strategy=strategy)

    trades_list = fetch_trades_list(base, auth)
    first_trade_ts = get_first_trade_timestamp(base, auth, prof, trades_list)
    last_trade_ts = get_last_trade_timestamp(trades_list, prof)

    wins = prof.get("winning_trades") or 0
    losses = prof.get("losing_trades") or 0
    closed = wins + losses

    pf = prof.get("profit_factor")
    mdd = prof.get("max_drawdown")
    mdd_pct = None if mdd is None else (mdd * 100 if isinstance(mdd, (int, float)) and abs(mdd) <= 1 else float(mdd))

    profit_all = prof.get("profit_all_percent")

    return make_row(
        name,
        port=port,
        bot=bot_name,
        strategy=strategy,
        trades=prof.get("trade_count") or 0,
        last_trade=format_since_timestamp(last_trade_ts),
        last_trade_ts=last_trade_ts,
        win_rate=(wins / closed * 100.0) if closed else None,
        profit_all=profit_all,
        profit_closed=prof.get("profit_closed_percent"),
        pf="-" if pf is None else f"{pf:.2f}",
        sharpe=calculate_annualized_sharpe(trades_list, cfg.get("timeframe")),
        max_dd=mdd_pct,
        days=days_since_first_trade(first_trade_ts),
        cagr=calculate_cagr(profit_all, first_trade_ts),
    )


def print_table(rows: List[Dict[str, Any]]) -> None:
    def sort_key(r: Dict[str, Any]):
        has_data = 0 if r["trades"] == "-" else 1
        pa = r.get("profit_all")
        return (has_data, pa if isinstance(pa, (int, float)) else float("-inf"), r.get("trades") or -1)

    rows = sorted(rows, key=sort_key, reverse=True)
    rendered = [[CELL_RENDERERS[key](r) for _, key in HEADERS] for r in rows]

    col_w = [
        max(len(title), *(plain_text_len(row[i]) for row in rendered)) if rendered else len(title)
        for i, (title, _) in enumerate(HEADERS)
    ]

    header_row = " | ".join(
        f"{Colors.BOLD}{Colors.WHITE}{title:<{col_w[i]}}{Colors.RESET}"
        for i, (title, _) in enumerate(HEADERS)
    )
    separator = "-+-".join("-" * w for w in col_w)
    print(header_row)
    print(f"{Colors.DIM}{separator}{Colors.RESET}")

    for row in rendered:
        print(" | ".join(text + " " * (col_w[i] - plain_text_len(text)) for i, text in enumerate(row)))


def main() -> None:
    auth = load_api_auth()

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}LightGBM Strategy - Performance Monitor{Colors.RESET}")
    print(f"{Colors.DIM}Searching for containers with '{CONTAINER_KEYWORD}' in the name...{Colors.RESET}\n")

    conts = [c for c in docker_containers() if CONTAINER_KEYWORD.lower() in c["name"].lower()]
    if not conts:
        print(f"{Colors.BRIGHT_RED}No containers with '{CONTAINER_KEYWORD}' in the name were found.{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Found {len(conts)} container(s) with '{CONTAINER_KEYWORD}' in the name{Colors.RESET}\n")

    rows = [
        container_row(c["name"], c["port"], auth) if c["port"] is not None else make_row(c["name"])
        for c in conts
    ]
    print_table(rows)

    print(
        f"\n{Colors.DIM}Legend:{Colors.RESET} "
        f"{Colors.BRIGHT_GREEN}Good{Colors.RESET} | {Colors.BRIGHT_RED}Bad{Colors.RESET} | {Colors.DIM}No Data{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Rules: Profit >0 good | Win Rate >=50% good | PF >=1.0 good | Sharpe >=1.0 good | CAGR >10% good{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Sharpe: annualized, per-bar returns with flat bars = 0 (same convention as src/backtest.py).{Colors.RESET}\n"
    )


if __name__ == "__main__":
    main()
