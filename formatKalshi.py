from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal

import pandas as pd


def _prompt(msg: str, default: str | None = None) -> str:
    if default is not None:
        s = input(f"{msg} [{default}]: ").strip()
        return s or default
    return input(f"{msg}: ").strip()


def _prompt_float(msg: str, default: float) -> float:
    s = _prompt(msg, str(default))
    try:
        return float(s)
    except ValueError:
        print(f"couldn't parse that, using {default}", file=sys.stderr)
        return default


def _ask_slice_start() -> pd.Timestamp:
    print()
    iso = _prompt("Slice starts at (ISO UTC, or leave empty to type date parts)", "")
    if iso:
        return pd.to_datetime(iso, utc=True)
    y = _prompt("Year", "")
    m = _prompt("Month", "")
    d = _prompt("Day", "")
    h = _prompt("Hour 0–23", "0")
    if not (y and m and d):
        print("need year, month, and day for a slice", file=sys.stderr)
        sys.exit(2)
    hh = int(h) if h else 0
    return pd.Timestamp(int(y), int(m), int(d), hour=hh, tz="UTC")


def run_interactive() -> None:
    if not sys.stdin.isatty():
        print("need a real terminal for prompts", file=sys.stderr)
        sys.exit(2)

    print("Turn hourly data into 12-hour rows (last price in each window, ÷100 on the price).\n")

    in_path = _prompt("Path to hourly dataset CSV", "")
    if not in_path:
        in_path = _prompt("try again — path to CSV", "")
    if not in_path:
        sys.exit("no input path")

    out_path = ""
    while not out_path:
        out_path = _prompt("Output CSV path")
        out_path = out_path + ".csv"

    print("\n1 = whole file, windows from first row in the file")
    print("2 = only keep a date range you pick (UTC), then same 12h logic from that start")
    mode = _prompt("1 or 2", "1").strip()

    slice_start = None
    slice_days = 7.0
    if mode == "2":
        slice_start = _ask_slice_start()
        slice_days = _prompt_float("how many days from that start", 7.0)
    elif mode != "1":
        print("didn't recognize that, doing whole file", file=sys.stderr)

    period_hours = _prompt_float("hours per bucket (usually 12)", 12.0)

    out = hourly_csv_to_12h(
        in_path,
        out_path,
        period_hours=period_hours,
        slice_start=slice_start,
        slice_days=slice_days,
    )
    tag = f"sliced {slice_days}d from {slice_start}" if slice_start else "full file"
    print(f"\n{len(out)} rows → {out_path} ({tag})")


def _scale_price(raw) -> float:
    d = Decimal(str(raw).strip())
    return float(d / Decimal("100"))


def _bucket_last(df: pd.DataFrame, raw_col: str, period_hours: float, anchor: pd.Timestamp | None):
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "yes_price"])

    step = pd.Timedelta(hours=period_hours)
    if anchor is None:
        anchor = df["timestamp"].iloc[0]
    last_ts = df["timestamp"].iloc[-1]
    rows = []
    w0 = anchor

    while w0 <= last_ts:
        w1 = w0 + step
        chunk = df[(df["timestamp"] >= w0) & (df["timestamp"] < w1)]
        if chunk.empty:
            w0 = w1
            continue
        tail = chunk.iloc[-1]
        rows.append({"timestamp": w0, "yes_price": _scale_price(tail[raw_col])})
        if tail["timestamp"] >= last_ts:
            break
        w0 = w1

    return pd.DataFrame(rows)


def _slice_from_args(args: argparse.Namespace) -> pd.Timestamp | None:
    if getattr(args, "from_datetime", None):
        return pd.to_datetime(args.from_datetime, utc=True)
    cal = (args.year, args.month, args.day)
    if any(x is not None for x in cal):
        if any(x is None for x in cal):
            print("--year, --month, --day all needed together", file=sys.stderr)
            sys.exit(2)
        h = 0 if args.hour is None else int(args.hour)
        return pd.Timestamp(int(args.year), int(args.month), int(args.day), hour=h, tz="UTC")
    if args.hour is not None and args.month is None:
        print("--hour needs --year --month --day", file=sys.stderr)
        sys.exit(2)
    return None


def hourly_csv_to_12h(
    input_path: str,
    output_path: str,
    period_hours: float = 12.0,
    slice_start: pd.Timestamp | None = None,
    slice_days: float = 7.0,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if df.empty:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(columns=["timestamp", "yes_price"]).to_csv(output_path, index=False)
        return df

    if "timestamp" not in df.columns:
        raise ValueError(f"need a timestamp column in {input_path}")
    prices = [c for c in df.columns if c != "timestamp"]
    if not prices:
        raise ValueError("need one price column besides timestamp")
    if len(prices) > 1:
        raise ValueError(f"only one price column for now, got: {prices}")

    raw = prices[0]
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    anchor = None
    if slice_start is not None:
        end = slice_start + pd.Timedelta(days=slice_days)
        df = df[(df["timestamp"] >= slice_start) & (df["timestamp"] < end)]
        anchor = slice_start

    agg = _bucket_last(df, raw, period_hours, anchor)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    agg.to_csv(output_path, index=False)
    return agg


def main():
    if len(sys.argv) == 1 and sys.stdin.isatty():
        run_interactive()
        return

    p = argparse.ArgumentParser(description="Hourly CSV → 12h last-price bars, price ÷ 100.")
    p.add_argument("-i", "--interactive", action="store_true", help="prompts instead of flags")
    p.add_argument("--input", default="data/gas_kalshi.csv", help="hourly CSV")
    p.add_argument("--output", default="data/gas_kalshi_12h.csv", help="where to write")
    p.add_argument("--period-hours", type=float, default=12.0)
    p.add_argument("--from-datetime", default=None, metavar="ISO")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--month", type=int, default=None)
    p.add_argument("--day", type=int, default=None)
    p.add_argument("--hour", type=int, default=None)
    p.add_argument("--slice-days", type=float, default=7.0)
    args = p.parse_args()

    if args.interactive:
        run_interactive()
        return

    start = _slice_from_args(args)
    out = hourly_csv_to_12h(
        args.input,
        args.output,
        period_hours=args.period_hours,
        slice_start=start,
        slice_days=args.slice_days,
    )
    tag = f"slice {args.slice_days}d from {start}" if start else "full file"
    print(f"{len(out)} rows → {args.output} ({tag})")


if __name__ == "__main__":
    main()
