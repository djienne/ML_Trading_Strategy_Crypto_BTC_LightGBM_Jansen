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

import math

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable
from datetime import datetime, timezone

import requests
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "user_data" / "config.json"

# Read API credentials from project config
try:
    _cfg = json.loads(CONFIG_PATH.read_text())
    _api = _cfg.get("api_server", {})
    USERNAME = _api.get("username", "freqtrader")
    PASSWORD = _api.get("password", "")
except (FileNotFoundError, json.JSONDecodeError):
    USERNAME = "freqtrader"
    PASSWORD = ""

TIMEOUT = 3  # seconds
CONTAINER_KEYWORD = "lgbm"  # Filter containers containing this keyword

PORT_RE = re.compile(r"(?:\d{1,3}(?:\.\d{1,3}){3}:)?(\d+)->8080/tcp")

# ANSI Color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    # Text colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

def colorize_profit(value: Optional[float]) -> str:
    if value is None:
        return f"{Colors.DIM}-{Colors.RESET}"
    pct_str = f"{value:.2f}%"
    return f"{Colors.BRIGHT_GREEN}{pct_str}{Colors.RESET}" if value > 0 else f"{Colors.BRIGHT_RED}{pct_str}{Colors.RESET}"

def colorize_win_rate(value: Optional[float]) -> str:
    if value is None:
        return f"{Colors.DIM}-{Colors.RESET}"
    pct_str = f"{value:.2f}%"
    return f"{Colors.BRIGHT_GREEN}{pct_str}{Colors.RESET}" if value >= 50 else f"{Colors.BRIGHT_RED}{pct_str}{Colors.RESET}"

def colorize_profit_factor(pf_str: str) -> str:
    if pf_str == "-":
        return f"{Colors.DIM}-{Colors.RESET}"
    try:
        pf_val = float(pf_str)
        return f"{Colors.BRIGHT_GREEN}{pf_str}{Colors.RESET}" if pf_val >= 1.0 else f"{Colors.BRIGHT_RED}{pf_str}{Colors.RESET}"
    except ValueError:
        return f"{Colors.DIM}{pf_str}{Colors.RESET}"

def colorize_cagr(value: Optional[float]) -> str:
    if value is None:
        return f"{Colors.DIM}-{Colors.RESET}"
    pct_str = f"{value:.2f}%"
    return f"{Colors.BRIGHT_GREEN}{pct_str}{Colors.RESET}" if value > 10 else f"{Colors.BRIGHT_RED}{pct_str}{Colors.RESET}"

def colorize_sharpe(value: Optional[float]) -> str:
    if value is None:
        return f"{Colors.DIM}-{Colors.RESET}"
    val_str = f"{value:.2f}"
    return f"{Colors.BRIGHT_GREEN}{val_str}{Colors.RESET}" if value >= 1.0 else f"{Colors.BRIGHT_RED}{val_str}{Colors.RESET}"

def colorize_drawdown(value: Optional[float]) -> str:
    if value is None:
        return f"{Colors.DIM}-{Colors.RESET}"
    pct_str = f"{value:.2f}%"
    return f"{Colors.WHITE}{pct_str}{Colors.RESET}" if value == 0 else f"{Colors.BRIGHT_RED}{pct_str}{Colors.RESET}"

def colorize_trades(trades: Any) -> str:
    if trades == "-":
        return f"{Colors.DIM}-{Colors.RESET}"
    try:
        return f"{int(trades)}"
    except (ValueError, TypeError):
        return f"{Colors.DIM}{trades}{Colors.RESET}"

def colorize_last_trade(value: Optional[str], timestamp_ms: Optional[int]) -> str:
    if not value or timestamp_ms is None:
        return f"{Colors.DIM}-{Colors.RESET}"
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

def colorize_container_name(name: str) -> str:
    return f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{name}{Colors.RESET}"

def colorize_port(port: Any) -> str:
    if port == "-":
        return f"{Colors.DIM}-{Colors.RESET}"
    return f"{Colors.BRIGHT_MAGENTA}{port}{Colors.RESET}"

def colorize_strategy(strategy: str) -> str:
    if strategy == "-":
        return f"{Colors.DIM}-{Colors.RESET}"
    return f"{Colors.YELLOW}{strategy}{Colors.RESET}"

def colorize_bot_name(bot: str) -> str:
    if bot == "-":
        return f"{Colors.DIM}-{Colors.RESET}"
    return f"{Colors.CYAN}{bot}{Colors.RESET}"

def colorize_days(days: Optional[int]) -> str:
    if days is None:
        return f"{Colors.DIM}-{Colors.RESET}"
    return f"{days}"

# ---- Robust date parsing helpers ----

def try_parse_dt(val: Any) -> Optional[int]:
    """
    Parse various datetime representations commonly returned by Freqtrade.
    Returns milliseconds since epoch (UTC) or None.
    Accepts ISO strings (with/without 'Z'), epoch seconds/ms (int/float), or dicts with common keys.
    """
    if val is None:
        return None
    # numeric
    if isinstance(val, (int, float)):
        if val > 1e12:      # ms
            return int(val)
        if val > 1e9:       # s
            return int(val * 1000)
        return None
    # string
    if isinstance(val, str):
        s = val.strip()
        try:
            if s.isdigit():
                return try_parse_dt(int(s))
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    # dict or other: try common keys
    if isinstance(val, dict):
        for k in ("open_date", "open_at", "open_time", "opened_at", "date_open", "date"):
            ts = try_parse_dt(val.get(k))
            if ts:
                return ts
    return None

def extract_first_ts_from_any(obj: Any, keys: Iterable[str]) -> Optional[int]:
    """
    Search dict OR list-of-dicts for the first parseable timestamp in given keys.
    """
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

def get_first_trade_timestamp(base: str, auth: HTTPBasicAuth, prof: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    Robustly find the timestamp (ms) of the earliest trade (open or closed).
    Try, in order:
      1) /status (dict OR list) fields like 'first_trade_date' or 'first_trade_timestamp'
      2) hints on /profit
      3) /trades (any shape) → earliest open date across items
      4) /closed_trades then /open_trades (fallbacks)
    """
    # 1) /status
    status = get_json(f"{base}/status", auth)
    ts = extract_first_ts_from_any(status, ("first_trade_date", "first_trade_timestamp", "first_trade"))
    if ts:
        return ts

    # 2) /profit hints
    if isinstance(prof, dict):
        ts = extract_first_ts_from_any(prof, ("first_trade_date", "first_trade_timestamp"))
        if ts:
            return ts

    # 3) /trades
    for url in (
        f"{base}/trades?limit=5000",  # big net to find earliest
        f"{base}/trades",
    ):
        data = get_json(url, auth)
        if data:
            trades_list = (data.get("trades") or data.get("data") or data) if isinstance(data, (dict, list)) else []
            if not isinstance(trades_list, list):
                trades_list = []
            ts = extract_earliest_open_ts_from_trades(trades_list)
            if ts:
                return ts

    # 4) explicit closed/open endpoints
    for endpoint in ("closed_trades", "open_trades"):
        data = get_json(f"{base}/{endpoint}", auth)
        if data:
            items = (data.get(endpoint) or data.get("data") or data) if isinstance(data, (dict, list)) else []
            if not isinstance(items, list):
                items = []
            ts = extract_earliest_open_ts_from_trades(items)
            if ts:
                return ts

    return None

def get_last_trade_timestamp(base: str, auth: HTTPBasicAuth, prof: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    Find the newest known trade timestamp in milliseconds.
    Prefer full trade/order details, then fall back to /profit's latest trade hint.
    """
    for url in (
        f"{base}/trades?limit=5000",
        f"{base}/trades",
    ):
        data = get_json(url, auth)
        if data:
            trades_list = (data.get("trades") or data.get("data") or data) if isinstance(data, (dict, list)) else []
            if not isinstance(trades_list, list):
                trades_list = []
            ts = extract_latest_ts_from_trades(trades_list)
            if ts:
                return ts

    if isinstance(prof, dict):
        return extract_first_ts_from_any(
            prof,
            ("latest_trade_timestamp", "latest_trade_date", "latest_trade"),
        )

    return None

def calculate_cagr(profit_all_percent: Optional[float], first_trade_timestamp_ms: Optional[int]) -> Optional[float]:
    """
    CAGR from current PnL of ALL trades (profit_all_percent) and time since first trade.
    Annualizes even for short histories. Returns % or None if inputs invalid.
    """
    if profit_all_percent is None or first_trade_timestamp_ms is None:
        return None
    try:
        first_trade_date = datetime.fromtimestamp(first_trade_timestamp_ms / 1000, tz=timezone.utc)
        current_date = datetime.now(timezone.utc)
        elapsed_days = (current_date - first_trade_date).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return None

        years_elapsed = elapsed_days / 365.25
        ending_value = 100.0 + float(profit_all_percent)
        if ending_value <= 0:
            return None  # nuked account case

        cagr = (pow(ending_value / 100.0, 1.0 / years_elapsed) - 1.0) * 100.0
        return cagr
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

def fetch_trades_list(base: str, auth: HTTPBasicAuth) -> List[Dict[str, Any]]:
    for url in (f"{base}/trades?limit=5000", f"{base}/trades"):
        data = get_json(url, auth)
        if data:
            trades_list = (data.get("trades") or data.get("data") or data) if isinstance(data, (dict, list)) else []
            if isinstance(trades_list, list) and trades_list:
                return trades_list
    return []

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
        current_date = datetime.now(timezone.utc)
        elapsed_days = (current_date - first_trade_date).total_seconds() / 86400.0
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

def pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.2f}%"

def main() -> None:
    auth = HTTPBasicAuth(USERNAME, PASSWORD)

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}LightGBM Strategy - Performance Monitor{Colors.RESET}")
    print(f"{Colors.DIM}Searching for containers with '{CONTAINER_KEYWORD}' in the name...{Colors.RESET}\n")

    conts = [c for c in docker_containers() if CONTAINER_KEYWORD.lower() in c["name"].lower()]
    if not conts:
        print(f"{Colors.BRIGHT_RED}No containers with '{CONTAINER_KEYWORD}' in the name were found.{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Found {len(conts)} container(s) with '{CONTAINER_KEYWORD}' in the name{Colors.RESET}\n")

    rows: List[Dict[str, Any]] = []
    for c in conts:
        name, port = c["name"], c["port"]
        strategy = "-"
        bot_name = "-"

        if port is None:
            rows.append(
                {
                    "container": name,
                    "port": "-",
                    "bot": bot_name,
                    "strategy": strategy,
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
            )
            continue

        base = f"http://127.0.0.1:{port}/api/v1"
        cfg = get_json(f"{base}/show_config", auth) or {}
        bot_name = cfg.get("bot_name") or "-"
        strategy = cfg.get("strategy") or "-"

        prof = get_json(f"{base}/profit", auth)
        if not prof:
            rows.append(
                {
                    "container": name,
                    "port": port,
                    "bot": bot_name,
                    "strategy": strategy,
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
            )
            continue

        first_trade_ts = get_first_trade_timestamp(base, auth, prof)
        last_trade_ts = get_last_trade_timestamp(base, auth, prof)
        last_trade = format_since_timestamp(last_trade_ts)

        w = (prof.get("winning_trades") or 0) if isinstance(prof, dict) else 0
        l = (prof.get("losing_trades") or 0) if isinstance(prof, dict) else 0
        tc = (prof.get("trade_count") or 0) if isinstance(prof, dict) else 0
        closed = w + l
        win_rate = (w / closed * 100.0) if closed else None

        pf = (prof.get("profit_factor") if isinstance(prof, dict) else None)
        pf_str = "-" if pf is None else f"{pf:.2f}"

        mdd = (prof.get("max_drawdown") if isinstance(prof, dict) else None)
        mdd_pct = None if mdd is None else (mdd * 100 if isinstance(mdd, (int, float)) and abs(mdd) <= 1 else float(mdd))

        profit_all = (prof.get("profit_all_percent") if isinstance(prof, dict) else None)
        cagr = calculate_cagr(profit_all, first_trade_ts)
        days = days_since_first_trade(first_trade_ts)
        sharpe = calculate_annualized_sharpe(
            fetch_trades_list(base, auth),
            cfg.get("timeframe"),
        )

        rows.append(
            {
                "container": name,
                "port": port,
                "bot": bot_name,
                "strategy": strategy,
                "trades": tc,
                "last_trade": last_trade,
                "last_trade_ts": last_trade_ts,
                "win_rate": win_rate,
                "profit_all": profit_all,
                "profit_closed": prof.get("profit_closed_percent") if isinstance(prof, dict) else None,
                "pf": pf_str,
                "sharpe": sharpe,
                "max_dd": mdd_pct,
                "days": days,
                "cagr": cagr,
            }
        )

    headers = [
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

    def cell(key: str, r: Dict[str, Any]) -> str:
        v = r.get(key)
        if key == "container":
            return colorize_container_name(str(v))
        if key == "port":
            return colorize_port(v)
        if key == "bot":
            return colorize_bot_name(str(v))
        if key == "strategy":
            return colorize_strategy(str(v))
        if key == "trades":
            return colorize_trades(v)
        if key == "last_trade":
            return colorize_last_trade(v, r.get("last_trade_ts"))
        if key == "win_rate":
            return colorize_win_rate(v)
        if key in ("profit_all", "profit_closed"):
            return colorize_profit(v)
        if key == "max_dd":
            return colorize_drawdown(v)
        if key == "pf":
            return colorize_profit_factor(str(v))
        if key == "sharpe":
            return colorize_sharpe(v)
        if key == "days":
            return colorize_days(v)
        if key == "cagr":
            return colorize_cagr(v)
        return str(v)

    # strip ANSI to compute widths
    def plain_text_len(text: str) -> int:
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return len(ansi_escape.sub('', text))

    # widths
    col_w: Dict[str, int] = {}
    for title, key in headers:
        w = len(title)
        for r in rows:
            cell_content = cell(key, r)
            w = max(w, plain_text_len(cell_content))
        col_w[title] = w

    # header
    header_row = " | ".join(f"{Colors.BOLD}{Colors.WHITE}{t:<{col_w[t]}}{Colors.RESET}" for t, _ in headers)
    separator = "-+-".join("-" * col_w[t] for t, _ in headers)
    print(header_row)
    print(f"{Colors.DIM}{separator}{Colors.RESET}")

    # sort
    def sort_key(r: Dict[str, Any]):
        has_data = 0 if r["trades"] == "-" else 1
        pa = r.get("profit_all")
        return (has_data, pa if isinstance(pa, (int, float)) else float("-inf"), r.get("trades") or -1)

    rows.sort(key=sort_key, reverse=True)

    # rows
    for r in rows:
        row_cells = []
        for title, key in headers:
            cell_content = cell(key, r)
            padding = col_w[title] - plain_text_len(cell_content)
            row_cells.append(cell_content + " " * padding)
        print(" | ".join(row_cells))

    print(
        f"\n{Colors.DIM}Legend:{Colors.RESET} "
        f"{Colors.BRIGHT_GREEN}Good{Colors.RESET} | {Colors.BRIGHT_RED}Bad{Colors.RESET} | {Colors.DIM}No Data{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Rules: Profit >0 good | Win Rate >=50% good | PF >=1.0 good | Sharpe >=1.0 good | CAGR >10% good{Colors.RESET}\n"
    )
    print(
        f"{Colors.DIM}Sharpe: annualized, per-bar returns with flat bars = 0 (same convention as src/backtest.py).{Colors.RESET}\n"
    )

if __name__ == "__main__":
    main()
