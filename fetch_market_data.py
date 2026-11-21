import datetime as dt
import io
import json
import math
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OUTPUT_PATH = Path("data.json")
TAIPEI = ZoneInfo("Asia/Taipei")

MARKETS = {
    "TWII": "^TWII",
    "TSMC": "2330.TW",
    "SOX": "^SOX",
    "NVDA": "NVDA",
    "IXIC": "^IXIC",
}

TECH_SYMBOLS = ["NVDA", "TSLA", "AMD", "SMCI"]

TWSE_ENDPOINTS = [
    "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
    "https://www.twse.com.tw/exchangeReport/MI_MARGN",
]

TREASURY_BASE = (
    "https://home.treasury.gov/resource-center/"
    "data-chart-center/interest-rates/"
    "daily-treasury-rates.csv"
)


def now():
    return dt.datetime.now(TAIPEI).isoformat(timespec="seconds")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def num(value):
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if not value or value in {"-", "--", "N/A"}:
            return None

    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def positive(value):
    value = num(value)
    return value if value is not None and value > 0 else None


def nonnegative(value):
    value = num(value)
    return value if value is not None and value >= 0 else None


def rounded(value, digits=2):
    value = num(value)
    return None if value is None else round(value, digits)


def parse_date(value):
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(str(value), pattern).date()
        except (TypeError, ValueError):
            continue
    return None


def unix_date(value):
    value = positive(value)
    if value is None:
        return None

    try:
        return dt.datetime.fromtimestamp(
            value, dt.timezone.utc
        ).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def load_previous():
    if not OUTPUT_PATH.exists():
        return {}

    try:
        with OUTPUT_PATH.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("JSON 最外層必須是物件")
        return data
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"舊 data.json 無法讀取，為避免覆寫歷史資料而停止：{error}"
        ) from error


def save_output(data):
    temporary = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")
        temporary.replace(OUTPUT_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def session_factory():
    session = requests.Session()

    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=False,
    )

    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "MarketDashboard/1.0",
        "Accept": "*/*",
    })

    return session


def valid_snapshot(item, fields, require_date=True):
    if not isinstance(item, dict):
        return False

    if item.get("status") not in {"ok", "partial", "cached"}:
        return False

    label = str(item.get("data_status", "")).lower()
    if any(word in label for word in ("sample", "mock", "demo")):
        return False

    if require_date and parse_date(item.get("data_date")) is None:
        return False

    return all(num(item.get(field)) is not None for field in fields)


def select_snapshot(current, previous, fields, attempted_at):
    previous = previous if isinstance(previous, dict) else {}
    current_ok = valid_snapshot(current, fields)
    previous_ok = valid_snapshot(previous, fields)

    if current_ok and previous_ok:
        if parse_date(current["data_date"]) < parse_date(
            previous["data_date"]
        ):
            current_ok = False
            current["note"] = "來源回傳較舊日期，保留既有快照。"

    if current_ok:
        return {
            **current,
            "is_cached": False,
            "retrieved_at": attempted_at,
            "last_attempt_at": attempted_at,
            "last_success_at": attempted_at,
            "last_error": None,
        }

    error = current.get("note") or "未取得有效資料"

    if previous_ok:
        return {
            **previous,
            "status": "cached",
            "is_cached": True,
            "last_attempt_at": attempted_at,
            "last_error": error,
        }

    return {
        **current,
        "status": "unavailable",
        "is_cached": False,
        "last_attempt_at": attempted_at,
        "last_success_at": None,
        "last_error": error,
    }


def fetch_markets(previous):
    performance = {}
    histories = {}
    old_performance = previous.get("market_performance") or {}
    old_histories = previous.get("normalized_history") or {}

    for key, symbol in MARKETS.items():
        attempted_at = now()
        history = None

        try:
            frame = yf.Ticker(symbol).history(
                period="1mo",
                auto_adjust=False,
                timeout=20,
            )

            if frame.empty or "Close" not in frame:
                raise ValueError("沒有收盤價資料")

            closes = pd.to_numeric(
                frame["Close"], errors="coerce"
            ).dropna()

            closes = closes[
                closes.map(lambda value: positive(value) is not None)
            ]

            if len(closes) < 2:
                raise ValueError("有效收盤價不足兩筆")

            latest = float(closes.iloc[-1])
            prior = float(closes.iloc[-2])
            first = float(closes.iloc[0])
            data_date = closes.index[-1].strftime("%Y-%m-%d")

            current = {
                "symbol": symbol,
                "price": rounded(latest),
                "daily_pct": rounded((latest / prior - 1) * 100),
                "month_pct": rounded((latest / first - 1) * 100),
                "data_date": data_date,
                "last_trade_date": data_date,
                "period_start": closes.index[0].strftime("%Y-%m-%d"),
                "source": "Yahoo Finance via yfinance",
                "status": "ok",
                "note": "未調整收盤價；近月為抓取區間首尾比較。",
            }

            history = {
                "dates": [
                    date.strftime("%Y-%m-%d")
                    for date in closes.index
                ],
                "values": [
                    rounded(value / first * 100)
                    for value in closes
                ],
                "data_date": data_date,
                "status": "ok",
                "is_cached": False,
            }

        except Exception as error:
            current = {
                "symbol": symbol,
                "price": None,
                "daily_pct": None,
                "month_pct": None,
                "data_date": None,
                "status": "unavailable",
                "note": str(error),
            }

        selected = select_snapshot(
            current,
            old_performance.get(key),
            ["price", "daily_pct", "month_pct"],
            attempted_at,
        )

        performance[key] = selected

        if selected.get("status") == "ok" and history is not None:
            histories[key] = history

        elif selected.get("status") == "cached":
            old_history = old_histories.get(key)
            if isinstance(old_history, dict):
                histories[key] = {
                    **old_history,
                    "status": "cached",
                    "is_cached": True,
                }

        log(f"行情 {key}: {selected.get('status')}")

    return performance, histories


def parse_margin(payload):
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        raise ValueError(
            f"TWSE 狀態非 OK："
            f"{payload.get('stat') if isinstance(payload, dict) else '非物件'}"
        )

    data_date = parse_date(payload.get("date"))
    if data_date is None:
        raise ValueError("來源缺少有效資料日期")

    tables = payload.get("tables")
    if not isinstance(tables, list):
        tables = [payload]

    for table in tables:
        if not isinstance(table, dict):
            continue

        fields = [str(value).strip() for value in table.get("fields", [])]
        if "前日餘額" not in fields or "今日餘額" not in fields:
            continue

        old_index = fields.index("前日餘額")
        new_index = fields.index("今日餘額")

        for row in table.get("data", []):
            if not isinstance(row, list) or not row:
                continue

            label = str(row[0]).replace(" ", "")
            if "融資金額" not in label:
                continue

            if "仟元" not in label and "千元" not in label:
                raise ValueError(f"無法辨識融資金額單位：{label}")

            if len(row) <= max(old_index, new_index):
                raise ValueError("融資欄位不足")

            old_balance = nonnegative(row[old_index])
            new_balance = nonnegative(row[new_index])

            if old_balance is None or new_balance is None:
                raise ValueError("融資餘額不是有效數值")

            return {
                # 沿用前端既有鍵名，實際單位為億元。
                "balance_billion": rounded(new_balance / 100000),
                "daily_diff_billion": rounded(
                    (new_balance - old_balance) / 100000
                ),
                "est_maintenance_ratio": None,
                "unit": "億元",
                "data_date": data_date.isoformat(),
                "source": "TWSE 信用交易統計",
                "status": "ok",
                "note": (
                    "上市市場資料，原始金額單位為仟元。"
                    "未取得獨立融資維持率來源，因此不估算。"
                ),
            }

    raise ValueError("未找到融資金額統計列")


def fetch_margin(session, previous):
    attempted_at = now()
    errors = []
    current = None

    for endpoint in TWSE_ENDPOINTS:
        try:
            response = session.get(
                endpoint,
                params={"response": "json", "selectType": "ALL"},
                headers={"Referer": "https://www.twse.com.tw/"},
                timeout=(10, 25),
            )
            response.raise_for_status()

            try:
                payload = response.json()
            except ValueError as error:
                raise ValueError(
                    "來源未回傳 JSON；"
                    f"Content-Type={response.headers.get('Content-Type')}"
                ) from error

            current = parse_margin(payload)
            current["source_url"] = response.url
            break

        except Exception as error:
            errors.append(str(error))
            time.sleep(1)

    if current is None:
        current = {
            "balance_billion": None,
            "daily_diff_billion": None,
            "est_maintenance_ratio": None,
            "data_date": None,
            "status": "unavailable",
            "note": "；".join(errors),
        }

    result = select_snapshot(
        current, previous, ["balance_billion"], attempted_at
    )
    log(f"TWSE: {result.get('status')}")
    return result


def treasury_year(session, year):
    response = session.get(
        f"{TREASURY_BASE}/{year}/all",
        params={
            "type": "daily_treasury_yield_curve",
            "field_tdr_date_value": str(year),
            "page": "",
            "_format": "csv",
        },
        timeout=(10, 30),
    )
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in frame.columns
    ]

    columns = ["3 Mo", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]
    required = ["Date", *columns]

    if any(column not in frame for column in required):
        raise ValueError("Treasury CSV 缺少必要欄位")

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=required)

    for column in columns:
        frame = frame[frame[column].map(lambda value: num(value) is not None)]

    if frame.empty:
        raise ValueError(f"{year} 年無有效美債資料")

    return frame


def fetch_treasury(session, previous):
    attempted_at = now()

    try:
        year = dt.datetime.now(TAIPEI).year
        frames = []
        warnings = []

        for requested_year in (year, year - 1):
            if frames:
                latest_date = frames[0]["Date"].max()
                target_date = latest_date - pd.DateOffset(months=1)
                if frames[0]["Date"].min() <= target_date:
                    break

            try:
                frames.append(treasury_year(session, requested_year))
            except Exception as error:
                warnings.append(str(error))

        if not frames:
            raise ValueError("；".join(warnings))

        frame = (
            pd.concat(frames, ignore_index=True)
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
        )

        latest = frame.iloc[-1]
        target_date = latest["Date"] - pd.DateOffset(months=1)
        comparison_rows = frame[frame["Date"] <= target_date]

        columns = ["3 Mo", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]
        values = [rounded(latest[column]) for column in columns]

        comparison_values = []
        comparison_date = None

        if not comparison_rows.empty:
            comparison = comparison_rows.iloc[-1]
            comparison_values = [
                rounded(comparison[column]) for column in columns
            ]
            comparison_date = comparison["Date"].strftime("%Y-%m-%d")

        current = {
            "maturities": ["3M", "2Y", "5Y", "10Y", "30Y"],
            "current": values,
            "one_month_ago": comparison_values,
            "comparison_date": comparison_date,
            "spread_10y_2y": rounded(values[3] - values[1]),
            "data_date": latest["Date"].strftime("%Y-%m-%d"),
            "source": "U.S. Department of the Treasury",
            "status": "ok",
            "note": (
                "殖利率單位為%，利差為百分點。"
                "比較期取往前一個月當日或之前最近有效資料。"
            ),
        }

        if warnings:
            current["note"] += " 部分請求失敗：" + "；".join(warnings)

    except Exception as error:
        current = {
            "maturities": ["3M", "2Y", "5Y", "10Y", "30Y"],
            "current": [],
            "one_month_ago": [],
            "comparison_date": None,
            "spread_10y_2y": None,
            "data_date": None,
            "status": "unavailable",
            "note": str(error),
        }

    # 為共用快照檢查建立內部數值欄位。
    current_values = current.get("current") or []
    current["_valid_10y"] = (
        current_values[3] if len(current_values) == 5 else None
    )

    old = dict(previous) if isinstance(previous, dict) else {}
    old_values = old.get("current") or []
    old["_valid_10y"] = (
        old_values[3] if len(old_values) == 5 else None
    )

    result = select_snapshot(
        current, old, ["_valid_10y"], attempted_at
    )
    result.pop("_valid_10y", None)

    log(f"Treasury: {result.get('status')}")
    return result


def old_rows(block):
    if not isinstance(block, dict):
        return {}

    items = block.get("items")
    if not isinstance(items, list):
        return {}

    return {
        str(item["ticker"]).upper(): item
        for item in items
        if isinstance(item, dict) and item.get("ticker")
    }


def valid_auto_row(row, fields):
    return (
        isinstance(row, dict)
        and row.get("input_method") == "automatic"
        and row.get("source") == "Yahoo Finance via yfinance"
        and row.get("status") in {"ok", "partial", "cached"}
        and any(num(row.get(field)) is not None for field in fields)
    )


def fallback_row(ticker, previous, fields, attempted_at, error):
    if valid_auto_row(previous, fields):
        return {
            **previous,
            "status": "cached",
            "is_cached": True,
            "last_attempt_at": attempted_at,
            "last_error": str(error),
        }

    return {
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance",
        "input_method": "automatic",
        "status": "unavailable",
        "is_cached": False,
        "data_date": None,
        "retrieved_at": None,
        "last_attempt_at": attempted_at,
        "last_success_at": None,
        "last_error": str(error),
    }


def make_short_row(ticker, info, previous, attempted_at):
    shares = nonnegative(info.get("sharesShort"))
    ratio = nonnegative(info.get("shortPercentOfFloat"))
    days = nonnegative(info.get("shortRatio"))
    data_date = unix_date(info.get("dateShortInterest"))

    if all(value is None for value in (shares, ratio, days)):
        raise ValueError("來源未提供空單資料")

    old = previous if isinstance(previous, dict) else {}
    old_valid = valid_auto_row(
        old, ["shares_short", "short_float_pct", "days_to_cover"]
    )

    current_date = parse_date(data_date)
    previous_date = parse_date(old.get("data_date")) if old_valid else None

    if previous_date and (
        current_date is None or current_date < previous_date
    ):
        raise ValueError("本次空單日期缺失或較舊，保留既有快照")

    compare_shares = None
    compare_date = None

    if previous_date and current_date:
        if current_date > previous_date:
            compare_shares = nonnegative(old.get("shares_short"))
            compare_date = previous_date.isoformat()

        elif current_date == previous_date:
            saved_date = parse_date(old.get("comparison_data_date"))
            if saved_date and saved_date < current_date:
                compare_date = saved_date.isoformat()
                compare_shares = nonnegative(
                    old.get("comparison_shares_short")
                )

    trend = "無可比資料"

    if shares is not None and compare_shares is not None:
        if shares > compare_shares:
            trend = "較前次資料日增加"
        elif shares < compare_shares:
            trend = "較前次資料日減少"
        else:
            trend = "較前次資料日持平"

    return {
        "ticker": ticker,
        "shares_short": int(shares) if shares is not None else None,
        "short_float_pct": rounded(ratio * 100) if ratio is not None else None,
        "days_to_cover": rounded(days),
        "trend": trend,
        "comparison_data_date": compare_date,
        "comparison_shares_short": compare_shares,
        "data_date": data_date,
        "source": "Yahoo Finance via yfinance",
        "input_method": "automatic",
        "status": (
            "ok"
            if all(
                value is not None
                for value in (shares, ratio, days, data_date)
            )
            else "partial"
        ),
        "is_cached": False,
        "retrieved_at": attempted_at,
        "last_attempt_at": attempted_at,
        "last_success_at": attempted_at,
        "last_error": None,
        "note": (
            "來源日期未提供時，不以抓取日期替代。"
            "未平倉空單不等於每日放空成交量。"
        ),
    }


def make_target_row(ticker, stock, info, attempted_at):
    warnings = []

    try:
        targets = stock.analyst_price_targets
        if not isinstance(targets, dict):
            targets = {}
    except Exception as error:
        targets = {}
        warnings.append(str(error))

    def pick(*values):
        for value in values:
            parsed = positive(value)
            if parsed is not None:
                return parsed
        return None

    price = pick(
        targets.get("current"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )
    mean = pick(targets.get("mean"), info.get("targetMeanPrice"))
    low = pick(targets.get("low"), info.get("targetLowPrice"))
    high = pick(targets.get("high"), info.get("targetHighPrice"))
    median = pick(targets.get("median"), info.get("targetMedianPrice"))

    if all(value is None for value in (mean, low, high, median)):
        raise ValueError("來源未提供分析師目標價")

    count = nonnegative(info.get("numberOfAnalystOpinions"))
    recommendation = info.get("recommendationKey")

    return {
        "ticker": ticker,
        "currency": info.get("currency"),
        "reference_price": rounded(price),
        "target_mean": rounded(mean),
        "target_low": rounded(low),
        "target_high": rounded(high),
        "target_median": rounded(median),
        "target_upside_pct": (
            rounded((mean / price - 1) * 100)
            if mean is not None and price is not None
            else None
        ),
        "analyst_count": int(count) if count is not None else None,
        "recommendation": (
            recommendation if isinstance(recommendation, str) else None
        ),
        "data_date": None,
        "quote_data_date": unix_date(info.get("regularMarketTime")),
        "source": "Yahoo Finance via yfinance",
        "input_method": "automatic",
        "status": "ok" if mean is not None and price is not None else "partial",
        "is_cached": False,
        "retrieved_at": attempted_at,
        "last_attempt_at": attempted_at,
        "last_success_at": attempted_at,
        "last_error": None,
        "note": (
            "來源未提供可辨識的目標價彙總更新日期。"
            "抓取時間不等於報告日期；目標價空間不是保證報酬。"
            + (" 專用介面訊息：" + "；".join(warnings) if warnings else "")
        ),
    }


def rows_status(rows):
    usable = [
        row for row in rows
        if row.get("status") in {"ok", "partial", "cached"}
    ]

    if not usable:
        return "unavailable"
    if len(usable) == len(rows) and all(
        row.get("status") == "ok" for row in usable
    ):
        return "ok"
    if all(row.get("status") == "cached" for row in usable):
        return "cached"
    return "partial"


def fetch_tech(previous):
    old_shorts = old_rows(previous.get("short_interest_top_tech"))
    old_targets = old_rows(previous.get("analyst_targets"))
    shorts = []
    targets = []

    for ticker in TECH_SYMBOLS:
        attempted_at = now()
        stock = yf.Ticker(ticker)
        info_error = None

        try:
            info = stock.get_info()
            if not isinstance(info, dict):
                raise ValueError("股票資訊不是物件")
        except Exception as error:
            info = {}
            info_error = str(error)

        try:
            short = make_short_row(
                ticker, info, old_shorts.get(ticker), attempted_at
            )
        except Exception as error:
            message = str(error)
            if info_error:
                message += f"；股票資訊請求失敗：{info_error}"
            short = fallback_row(
                ticker,
                old_shorts.get(ticker),
                ["shares_short", "short_float_pct", "days_to_cover"],
                attempted_at,
                message,
            )

        try:
            target = make_target_row(ticker, stock, info, attempted_at)
        except Exception as error:
            message = str(error)
            if info_error:
                message += f"；股票資訊請求失敗：{info_error}"
            target = fallback_row(
                ticker,
                old_targets.get(ticker),
                ["target_mean", "target_low", "target_high", "target_median"],
                attempted_at,
                message,
            )

        shorts.append(short)
        targets.append(target)

        log(
            f"{ticker}: short={short.get('status')}, "
            f"targets={target.get('status')}"
        )
        time.sleep(1)

    return (
        {
            "source": "Yahoo Finance via yfinance",
            "status": rows_status(shorts),
            "items": shorts,
            "note": "資料日期與狀態以每筆股票為準。",
        },
        {
            "source": "Yahoo Finance via yfinance",
            "status": rows_status(targets),
            "items": targets,
            "note": "分析師彙總資料，不是特定投行的完整配置報告。",
        },
    )


def rate_monitor(treasury):
    values = {
        key: num(value)
        for key, value in zip(
            treasury.get("maturities") or [],
            treasury.get("current") or [],
        )
    }

    three_month = values.get("3M")
    two_year = values.get("2Y")
    ten_year = values.get("10Y")

    def spread(left, right):
        if left is None or right is None:
            return None
        return rounded(left - right)

    return {
        "title": "利率市場指標",
        "source": treasury.get("source"),
        "status": treasury.get("status", "unavailable"),
        "is_cached": bool(treasury.get("is_cached")),
        "data_date": treasury.get("data_date"),
        "last_success_at": treasury.get("last_success_at"),
        "last_attempt_at": treasury.get("last_attempt_at"),
        "last_error": treasury.get("last_error"),
        "yield_3m_pct": three_month,
        "yield_2y_pct": two_year,
        "yield_10y_pct": ten_year,
        "spread_10y_2y_pp": spread(ten_year, two_year),
        "spread_10y_3m_pp": spread(ten_year, three_month),
        "note": (
            "美債殖利率不是聯準會政策利率；"
            "本區不提供 FedWatch 機率。"
            "殖利率單位為%，利差單位為百分點。"
        ),
    }


def main():
    log("開始更新免費自動化市場儀表板")

    previous = load_previous()
    performance, histories = fetch_markets(previous)

    with session_factory() as session:
        margin = fetch_margin(
            session, previous.get("tw_margin", {})
        )
        treasury = fetch_treasury(
            session, previous.get("treasury_yields", {})
        )

    shorts, targets = fetch_tech(previous)

    output = {
        "schema_version": 4,
        "last_updated": now(),
        "market_performance": performance,
        "normalized_history": histories,
        "tw_margin": margin,
        "treasury_yields": treasury,
        "rate_market_monitor": rate_monitor(treasury),
        "short_interest_top_tech": shorts,
        "analyst_targets": targets,
    }

    save_output(output)

    log(
        "data.json 已安全寫入；"
        f"融資={margin.get('status')}；"
        f"美債={treasury.get('status')}；"
        f"空單={shorts.get('status')}；"
        f"目標價={targets.get('status')}"
    )


if __name__ == "__main__":
    main()
