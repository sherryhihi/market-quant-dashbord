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
    "SPX": "^GSPC",
    "IXIC": "^IXIC",
    "NVDA": "NVDA",
    "AMD": "AMD",
    "SOX": "^SOX",
    "VIX": "^VIX",
    "TWII": "^TWII",
    "TSMC": "2330.TW",
}

TECH_SYMBOLS = ["NVDA", "AMD", "TSLA", "SMCI"]

TWSE_ENDPOINTS = [
    "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
    "https://www.twse.com.tw/exchangeReport/MI_MARGN",
]

TREASURY_BASE = (
    "https://home.treasury.gov/resource-center/"
    "data-chart-center/interest-rates/"
    "daily-treasury-rates.csv"
)

TREASURY_COLS = [
    "3 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr",
    "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]

TREASURY_LABELS = [
    "3M", "1Y", "2Y", "3Y", "5Y",
    "7Y", "10Y", "20Y", "30Y",
]


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
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    except (TypeError, ValueError):
        return None


def positive(value):
    parsed = num(value)
    return parsed if parsed is not None and parsed > 0 else None


def nonnegative(value):
    parsed = num(value)
    return parsed if parsed is not None and parsed >= 0 else None


def rounded(value, digits=2):
    parsed = num(value)
    return None if parsed is None else round(parsed, digits)


def parse_date(value):
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(
                str(value), pattern
            ).date()

        except (TypeError, ValueError):
            continue

    return None


def unix_date(value):
    parsed = positive(value)

    if parsed is None:
        return None

    try:
        return dt.datetime.fromtimestamp(
            parsed, dt.timezone.utc
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
            f"舊 data.json 無法讀取，停止覆寫：{error}"
        ) from error


def save_output(data):
    temporary = OUTPUT_PATH.with_name(
        OUTPUT_PATH.name + ".tmp"
    )

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

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry),
    )

    session.headers.update({
        "User-Agent": "MarketDashboardUpdater/3.0",
        "Accept": "*/*",
    })

    return session


def allowed_snapshot(item):
    if not isinstance(item, dict):
        return False

    if item.get("status") not in {"ok", "partial", "cached"}:
        return False

    data_status = str(
        item.get("data_status", "")
    ).lower()

    return not any(
        word in data_status
        for word in ("sample", "mock", "demo")
    )


def valid_numeric_snapshot(item, fields):
    return (
        allowed_snapshot(item)
        and parse_date(item.get("data_date")) is not None
        and all(
            num(item.get(field)) is not None
            for field in fields
        )
    )


def choose_snapshot(
    current,
    previous,
    validator,
    attempted_at,
):
    previous = previous if isinstance(previous, dict) else {}

    current_ok = validator(current)
    previous_ok = validator(previous)

    if current_ok and previous_ok:
        current_date = parse_date(current.get("data_date"))
        previous_date = parse_date(previous.get("data_date"))

        if (
            current_date is not None
            and previous_date is not None
            and current_date < previous_date
        ):
            current_ok = False
            current["note"] = (
                "來源回傳較舊日期，保留既有快照。"
            )

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
            "last_error": str(error),
        }

    return {
        **current,
        "status": "unavailable",
        "is_cached": False,
        "last_attempt_at": attempted_at,
        "last_success_at": None,
        "last_error": str(error),
    }


def fetch_markets(previous):
    performance = {}
    histories = {}

    old_performance = (
        previous.get("market_performance") or {}
    )

    old_histories = (
        previous.get("normalized_history") or {}
    )

    for key, symbol in MARKETS.items():
        attempted_at = now()
        history = None

        try:
            frame = yf.Ticker(symbol).history(
                period="3mo",
                auto_adjust=False,
                timeout=20,
            )

            if frame.empty or "Close" not in frame:
                raise ValueError("沒有收盤價資料")

            closes = pd.to_numeric(
                frame["Close"],
                errors="coerce",
            ).dropna()

            closes = closes[
                closes.map(
                    lambda value: positive(value) is not None
                )
            ].sort_index()

            if len(closes) < 2:
                raise ValueError("有效收盤價不足兩筆")

            latest = float(closes.iloc[-1])
            prior = float(closes.iloc[-2])

            cutoff = (
                closes.index[-1] - pd.DateOffset(months=1)
            )

            month_closes = closes.loc[
                closes.index >= cutoff
            ]

            if len(month_closes) < 2:
                raise ValueError("近月有效資料不足兩筆")

            first = float(month_closes.iloc[0])

            return_20d = None
            start_20d_date = None

            # 20 個交易日間隔需要 21 筆價格。
            if len(closes) >= 21:
                start_20d = float(closes.iloc[-21])

                return_20d = rounded(
                    (latest / start_20d - 1) * 100
                )

                start_20d_date = (
                    closes.index[-21].strftime("%Y-%m-%d")
                )

            data_date = (
                closes.index[-1].strftime("%Y-%m-%d")
            )

            current = {
                "symbol": symbol,
                "price": rounded(latest),
                "daily_pct": rounded(
                    (latest / prior - 1) * 100
                ),
                "month_pct": rounded(
                    (latest / first - 1) * 100
                ),
                "period_20d_pct": return_20d,
                "period_20d_start": start_20d_date,
                "period_start": (
                    month_closes.index[0]
                    .strftime("%Y-%m-%d")
                ),
                "data_date": data_date,
                "last_trade_date": data_date,
                "source": "Yahoo Finance via yfinance",
                "status": "ok",
                "note": (
                    "未調整價格資料；"
                    "近月為所選近月區間首尾比較。"
                    "交易時段內最後一筆日線可能尚未定案，"
                    "不保證為即時行情。"
                ),
            }

            history = {
                "dates": [
                    date.strftime("%Y-%m-%d")
                    for date in month_closes.index
                ],
                "values": [
                    rounded(value / first * 100)
                    for value in month_closes
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
                "period_20d_pct": None,
                "data_date": None,
                "status": "unavailable",
                "note": str(error),
            }

        selected = choose_snapshot(
            current,
            old_performance.get(key),
            lambda item: valid_numeric_snapshot(
                item,
                ["price", "daily_pct", "month_pct"],
            ),
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

        log(
            f"行情 {key}: {selected.get('status')} "
            f"@ {selected.get('data_date')}"
        )

        time.sleep(0.5)

    return performance, histories


def parse_margin(payload):
    if (
        not isinstance(payload, dict)
        or payload.get("stat") != "OK"
    ):
        status = (
            payload.get("stat")
            if isinstance(payload, dict)
            else "非 JSON 物件"
        )

        raise ValueError(f"TWSE 狀態非 OK：{status}")

    data_date = parse_date(payload.get("date"))

    if data_date is None:
        raise ValueError("TWSE 缺少有效資料日期")

    tables = payload.get("tables")

    if not isinstance(tables, list):
        tables = [payload]

    for table in tables:
        if not isinstance(table, dict):
            continue

        fields = [
            str(value).strip()
            for value in table.get("fields", [])
        ]

        if (
            "前日餘額" not in fields
            or "今日餘額" not in fields
        ):
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
                raise ValueError(
                    f"無法辨識融資單位：{label}"
                )

            if len(row) <= max(old_index, new_index):
                raise ValueError("TWSE 融資欄位不足")

            old_balance = nonnegative(row[old_index])
            new_balance = nonnegative(row[new_index])

            if old_balance is None or new_balance is None:
                raise ValueError("融資餘額不是有效數值")

            return {
                "balance_billion": rounded(
                    new_balance / 100000
                ),
                "daily_diff_billion": rounded(
                    (new_balance - old_balance) / 100000
                ),
                "est_maintenance_ratio": None,
                "unit": "億元",
                "data_date": data_date.isoformat(),
                "source": "TWSE 信用交易統計",
                "status": "ok",
                "note": (
                    "上市市場融資金額；"
                    "原始仟元換算為億元。"
                    "未取得獨立融資維持率資料，故不估算。"
                ),
            }

    raise ValueError("找不到融資金額統計列")


def fetch_margin(session, previous):
    attempted_at = now()
    errors = []
    current = None

    for endpoint in TWSE_ENDPOINTS:
        try:
            response = session.get(
                endpoint,
                params={
                    "response": "json",
                    "selectType": "ALL",
                },
                headers={
                    "Referer": "https://www.twse.com.tw/"
                },
                timeout=(10, 25),
            )

            response.raise_for_status()
            current = parse_margin(response.json())
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

    result = choose_snapshot(
        current,
        previous,
        lambda item: (
            valid_numeric_snapshot(
                item, ["balance_billion"]
            )
            and nonnegative(
                item.get("balance_billion")
            ) is not None
        ),
        attempted_at,
    )

    log(
        f"融資: {result.get('status')} "
        f"@ {result.get('data_date')}"
    )

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

    required = ["Date", *TREASURY_COLS]

    if any(column not in frame for column in required):
        raise ValueError("Treasury CSV 缺少必要欄位")

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    for column in TREASURY_COLS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(subset=required)

    for column in TREASURY_COLS:
        frame = frame[
            frame[column].map(
                lambda value: num(value) is not None
            )
        ]

    if frame.empty:
        raise ValueError(f"{year} 年無有效美債資料")

    return frame


def yield_mapping(snapshot):
    if not isinstance(snapshot, dict):
        return {}

    labels = snapshot.get("maturities")
    values = snapshot.get("current")

    if (
        not isinstance(labels, list)
        or not isinstance(values, list)
        or len(labels) != len(values)
    ):
        return {}

    return {
        label: num(value)
        for label, value in zip(labels, values)
    }


def valid_treasury(snapshot):
    if not allowed_snapshot(snapshot):
        return False

    if parse_date(snapshot.get("data_date")) is None:
        return False

    values = yield_mapping(snapshot)

    # 兼容舊版五期限及新版九期限。
    return all(
        values.get(label) is not None
        for label in ("3M", "2Y", "10Y")
    )


def fetch_treasury(session, previous):
    attempted_at = now()

    try:
        year = dt.datetime.now(TAIPEI).year
        frames = []
        warnings = []

        for requested_year in (year, year - 1):
            if frames:
                latest_date = frames[0]["Date"].max()
                cutoff = (
                    latest_date - pd.DateOffset(months=1)
                )

                if frames[0]["Date"].min() <= cutoff:
                    break

            try:
                frames.append(
                    treasury_year(session, requested_year)
                )

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

        values = [
            rounded(latest[column])
            for column in TREASURY_COLS
        ]

        recent_curves = []

        for _, row in frame.tail(3).iloc[::-1].iterrows():
            recent_curves.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "values": [
                    rounded(row[column])
                    for column in TREASURY_COLS
                ],
            })

        cutoff = (
            latest["Date"] - pd.DateOffset(months=1)
        )

        comparison_rows = frame[
            frame["Date"] <= cutoff
        ]

        comparison_values = []
        comparison_date = None

        if not comparison_rows.empty:
            comparison = comparison_rows.iloc[-1]

            comparison_values = [
                rounded(comparison[column])
                for column in TREASURY_COLS
            ]

            comparison_date = (
                comparison["Date"].strftime("%Y-%m-%d")
            )

        mapping = dict(zip(TREASURY_LABELS, values))

        current = {
            "maturities": TREASURY_LABELS,
            "current": values,
            "recent_curves": recent_curves,
            "one_month_ago": comparison_values,
            "comparison_date": comparison_date,
            "spread_10y_2y": rounded(
                mapping["10Y"] - mapping["2Y"]
            ),
            "data_date": (
                latest["Date"].strftime("%Y-%m-%d")
            ),
            "source": "U.S. Department of the Treasury",
            "status": "ok",
            "note": (
                "殖利率單位為%，利差為百分點。"
                "比較期為往前一個月當日或之前最近有效資料。"
            ),
        }

        if warnings:
            current["note"] += (
                " 部分請求警告：" + "；".join(warnings)
            )

    except Exception as error:
        current = {
            "maturities": TREASURY_LABELS,
            "current": [],
            "recent_curves": [],
            "one_month_ago": [],
            "comparison_date": None,
            "spread_10y_2y": None,
            "data_date": None,
            "status": "unavailable",
            "note": str(error),
        }

    result = choose_snapshot(
        current,
        previous,
        valid_treasury,
        attempted_at,
    )

    log(
        f"美債: {result.get('status')} "
        f"@ {result.get('data_date')}"
    )

    return result


def summarize_rows(rows):
    usable = [
        row
        for row in rows
        if row.get("status") in {"ok", "partial", "cached"}
    ]

    if not usable:
        return "unavailable"

    if len(usable) == len(rows) and all(
        row.get("status") == "ok"
        for row in usable
    ):
        return "ok"

    if all(
        row.get("status") == "cached"
        for row in usable
    ):
        return "cached"

    return "partial"


def valid_target_row(row):
    return (
        allowed_snapshot(row)
        and row.get("source") == "Yahoo Finance via yfinance"
        and any(
            positive(row.get(field)) is not None
            for field in (
                "target_mean",
                "target_median",
                "target_low",
                "target_high",
            )
        )
    )


def fetch_tech_targets(previous):
    old_map = {}

    old_block = previous.get("analyst_targets")

    if isinstance(old_block, dict):
        old_items = old_block.get("items")

        if isinstance(old_items, list):
            for item in old_items:
                if isinstance(item, dict) and item.get("ticker"):
                    old_map[
                        str(item["ticker"]).upper()
                    ] = item

    rows = []

    for ticker in TECH_SYMBOLS:
        attempted_at = now()
        stock = yf.Ticker(ticker)

        try:
            info = {}
            targets = {}
            warnings = []

            try:
                raw_info = stock.get_info()

                if isinstance(raw_info, dict):
                    info = raw_info

            except Exception as error:
                warnings.append(
                    f"股票資訊介面失敗：{error}"
                )

            try:
                raw_targets = stock.analyst_price_targets

                if isinstance(raw_targets, dict):
                    targets = raw_targets

            except Exception as error:
                warnings.append(
                    f"目標價介面失敗：{error}"
                )

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

            mean = pick(
                targets.get("mean"),
                info.get("targetMeanPrice"),
            )

            median = pick(
                targets.get("median"),
                info.get("targetMedianPrice"),
            )

            low = pick(
                targets.get("low"),
                info.get("targetLowPrice"),
            )

            high = pick(
                targets.get("high"),
                info.get("targetHighPrice"),
            )

            if all(
                value is None
                for value in (mean, median, low, high)
            ):
                raise ValueError(
                    "無有效目標價資料。"
                    + "；".join(warnings)
                )

            analyst_count = nonnegative(
                info.get("numberOfAnalystOpinions")
            )

            recommendation = info.get(
                "recommendationKey"
            )

            if not isinstance(recommendation, str):
                recommendation = None

            rows.append({
                "ticker": ticker,
                "currency": info.get("currency"),
                "reference_price": rounded(price),
                "target_mean": rounded(mean),
                "target_median": rounded(median),
                "target_low": rounded(low),
                "target_high": rounded(high),
                "target_upside_pct": (
                    rounded((mean / price - 1) * 100)
                    if mean is not None and price is not None
                    else None
                ),
                "analyst_count": (
                    int(analyst_count)
                    if analyst_count is not None
                    else None
                ),
                "recommendation": recommendation,
                "quote_data_date": unix_date(
                    info.get("regularMarketTime")
                ),
                "data_date": None,
                "source": "Yahoo Finance via yfinance",
                "input_method": "automatic",
                "status": (
                    "ok"
                    if mean is not None and price is not None
                    else "partial"
                ),
                "is_cached": False,
                "retrieved_at": attempted_at,
                "last_attempt_at": attempted_at,
                "last_success_at": attempted_at,
                "last_error": None,
                "note": (
                    "來源未提供可辨識的目標價彙總更新日期。"
                    "抓取時間不等於報告日期。"
                    "目標價相對空間不是保證報酬。"
                    + (
                        " 部分請求警告："
                        + "；".join(warnings)
                        if warnings
                        else ""
                    )
                ),
            })

        except Exception as error:
            old_row = old_map.get(ticker)

            if valid_target_row(old_row):
                rows.append({
                    **old_row,
                    "status": "cached",
                    "is_cached": True,
                    "last_attempt_at": attempted_at,
                    "last_error": str(error),
                })

            else:
                rows.append({
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
                })

        log(
            f"目標價 {ticker}: "
            f"{rows[-1].get('status')}"
        )

        time.sleep(1)

    return {
        "source": "Yahoo Finance via yfinance",
        "status": summarize_rows(rows),
        "items": rows,
        "note": (
            "分析師彙總目標價，"
            "不是特定投行的完整研究報告。"
        ),
    }


def build_rate_monitor(treasury):
    values = yield_mapping(treasury)

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
        "status": treasury.get(
            "status", "unavailable"
        ),
        "is_cached": bool(
            treasury.get("is_cached")
        ),
        "data_date": treasury.get("data_date"),
        "last_attempt_at": treasury.get(
            "last_attempt_at"
        ),
        "last_success_at": treasury.get(
            "last_success_at"
        ),
        "last_error": treasury.get("last_error"),
        "yield_3m_pct": three_month,
        "yield_2y_pct": two_year,
        "yield_10y_pct": ten_year,
        "spread_10y_2y_pp": spread(
            ten_year, two_year
        ),
        "spread_10y_3m_pp": spread(
            ten_year, three_month
        ),
        "note": (
            "美債殖利率不是聯準會政策利率，"
            "也不是 FedWatch 機率。"
            "殖利率單位為%，利差單位為百分點。"
        ),
    }


def main():
    log("開始更新市場儀表板")

    previous = load_previous()

    performance, histories = fetch_markets(previous)

    with session_factory() as session:
        margin = fetch_margin(
            session,
            previous.get("tw_margin", {}),
        )

        treasury = fetch_treasury(
            session,
            previous.get("treasury_yields", {}),
        )

    targets = fetch_tech_targets(previous)

    output = {
        "schema_version": 5,
        "last_updated": now(),
        "market_performance": performance,
        "normalized_history": histories,
        "tw_margin": margin,
        "treasury_yields": treasury,
        "rate_market_monitor": build_rate_monitor(
            treasury
        ),
        "analyst_targets": targets,

        # 與前一版 HTML 相容。
        # 此版不再抓取或展示空單資料。
        "short_interest_top_tech": {
            "source": None,
            "status": "disabled",
            "items": [],
            "note": "本版本暫停空單資料功能。",
        },
    }

    save_output(output)

    log(
        "data.json 已安全寫入；"
        f"融資={margin.get('status')}；"
        f"美債={treasury.get('status')}；"
        f"目標價={targets.get('status')}"
    )


if __name__ == "__main__":
    main()
