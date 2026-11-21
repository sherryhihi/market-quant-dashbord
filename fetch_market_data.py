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

TICKERS = {
    "TWII": "^TWII",
    "TSMC": "2330.TW",
    "SOX": "^SOX",
    "NVDA": "NVDA",
    "IXIC": "^IXIC",
}

TWSE_ENDPOINTS = [
    "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
    "https://www.twse.com.tw/exchangeReport/MI_MARGN",
]

TREASURY_BASE_URL = (
    "https://home.treasury.gov/resource-center/"
    "data-chart-center/interest-rates/"
    "daily-treasury-rates.csv"
)


def taipei_now():
    return dt.datetime.now(TAIPEI).isoformat(timespec="seconds")


def log(message):
    print(f"[{taipei_now()}] {message}", flush=True)


def build_session():
    session = requests.Session()

    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "MarketDashboard/1.0",
        "Accept": "*/*",
    })

    return session


def number(value):
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = value.replace(",", "").strip()

        if value in {"", "-", "--", "N/A", "nan"}:
            return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def parse_date(value):
    text = str(value or "").strip()

    for date_format in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, date_format).date()
        except ValueError:
            continue

    return None


def load_previous_data():
    if not OUTPUT_PATH.exists():
        return {}

    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("data.json 最外層必須是物件")

        return data

    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"無法讀取既有 data.json，停止更新以保護歷史資料：{error}"
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


def valid_snapshot(snapshot, required_fields):
    if not isinstance(snapshot, dict):
        return False

    status = str(snapshot.get("status", "")).lower()
    data_status = str(snapshot.get("data_status", "")).lower()

    if status in {"unavailable", "failed", "error", "sample"}:
        return False

    if any(
        marker in data_status
        for marker in ("sample", "mock", "demo", "unverified")
    ):
        return False

    if parse_date(snapshot.get("data_date")) is None:
        return False

    return all(
        number(snapshot.get(field)) is not None
        for field in required_fields
    )


def select_snapshot(current, previous, required_fields, attempted_at):
    """
    選擇本次有效快照；失敗時沿用整份舊快照，
    不混合不同交易日的資料。
    """
    current = current if isinstance(current, dict) else {}
    previous = previous if isinstance(previous, dict) else {}

    current_valid = (
        current.get("status") in {"ok", "partial"}
        and valid_snapshot(current, required_fields)
    )

    previous_valid = valid_snapshot(previous, required_fields)

    if current_valid and previous_valid:
        current_date = parse_date(current["data_date"])
        previous_date = parse_date(previous["data_date"])

        if current_date < previous_date:
            current_valid = False
            current = {
                **current,
                "note": (
                    f"來源回傳較舊資料日 {current['data_date']}，"
                    f"保留既有資料日 {previous['data_date']}。"
                ),
            }

    if current_valid:
        return {
            **current,
            "is_cached": False,
            "last_attempt_at": attempted_at,
            "last_success_at": attempted_at,
            "last_error": None,
        }

    error_message = (
        current.get("note")
        or current.get("last_error")
        or "未取得有效資料"
    )

    if previous_valid:
        result = {
            **previous,
            "status": "cached",
            "is_cached": True,
            "last_attempt_at": attempted_at,
            "last_error": str(error_message),
        }

        result.setdefault("last_success_at", None)
        return result

    return {
        **current,
        "status": "unavailable",
        "is_cached": False,
        "last_attempt_at": attempted_at,
        "last_success_at": None,
        "last_error": str(error_message),
    }


def fetch_indices_and_stocks(previous):
    performance = {}
    history_series = {}

    previous_performance = previous.get("market_performance") or {}
    previous_history = previous.get("normalized_history") or {}

    for name, symbol in TICKERS.items():
        attempted_at = taipei_now()
        current = {}
        current_history = None

        try:
            frame = yf.Ticker(symbol).history(
                period="1mo",
                auto_adjust=False,
                timeout=20,
            )

            if frame.empty or "Close" not in frame.columns:
                raise ValueError("沒有價格資料")

            closes = pd.to_numeric(
                frame["Close"], errors="coerce"
            ).dropna()

            closes = closes[
                closes.map(lambda value: math.isfinite(float(value)))
            ]

            if len(closes) < 2:
                raise ValueError("有效收盤價不足兩筆")

            latest = float(closes.iloc[-1])
            previous_close = float(closes.iloc[-2])
            first_close = float(closes.iloc[0])

            if min(latest, previous_close, first_close) <= 0:
                raise ValueError("價格必須大於零")

            data_date = closes.index[-1].strftime("%Y-%m-%d")

            current = {
                "symbol": symbol,
                "price": round(latest, 2),
                "daily_pct": round(
                    (latest / previous_close - 1) * 100, 2
                ),
                "month_pct": round(
                    (latest / first_close - 1) * 100, 2
                ),
                "data_date": data_date,
                "last_trade_date": data_date,
                "period_start": closes.index[0].strftime("%Y-%m-%d"),
                "return_basis": "未調整收盤價；近月為抓取區間首尾比較",
                "source": "Yahoo Finance via yfinance",
                "status": "ok",
            }

            normalized = closes / first_close * 100

            current_history = {
                "dates": [
                    date.strftime("%Y-%m-%d")
                    for date in closes.index
                ],
                "values": [
                    round(float(value), 2)
                    for value in normalized
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
                "source": "Yahoo Finance via yfinance",
                "status": "unavailable",
                "note": str(error),
            }

        selected = select_snapshot(
            current=current,
            previous=previous_performance.get(name),
            required_fields=["price", "daily_pct", "month_pct"],
            attempted_at=attempted_at,
        )

        performance[name] = selected

        if (
            selected.get("status") == "ok"
            and current_history is not None
        ):
            history_series[name] = current_history

        elif selected.get("status") == "cached":
            old_history = previous_history.get(name)

            if isinstance(old_history, dict):
                history_series[name] = {
                    **old_history,
                    "status": "cached",
                    "is_cached": True,
                }

        log(
            f"{name}: status={selected.get('status')}, "
            f"data_date={selected.get('data_date')}"
        )

    return performance, history_series


def parse_twse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("TWSE 回應不是 JSON 物件")

    if payload.get("stat") != "OK":
        raise ValueError(
            f"TWSE 回應狀態：{payload.get('stat', '未提供')}"
        )

    data_date = parse_date(payload.get("date"))

    if data_date is None:
        raise ValueError("TWSE 未提供可辨識的市場資料日期")

    tables = payload.get("tables")

    # 相容舊版直接使用 fields/data 的結構。
    if not isinstance(tables, list):
        tables = [payload]

    for table in tables:
        if not isinstance(table, dict):
            continue

        fields = [
            str(field).strip()
            for field in table.get("fields", [])
        ]

        rows = table.get("data", [])

        if not fields or not isinstance(rows, list):
            continue

        if "前日餘額" not in fields or "今日餘額" not in fields:
            continue

        previous_index = fields.index("前日餘額")
        current_index = fields.index("今日餘額")

        for row in rows:
            if not isinstance(row, list) or not row:
                continue

            label = str(row[0]).replace(" ", "")

            if "融資金額" not in label:
                continue

            if "仟元" not in label and "千元" not in label:
                raise ValueError(
                    f"無法確認融資金額單位：{label}"
                )

            if len(row) <= max(previous_index, current_index):
                raise ValueError("TWSE 融資資料欄位不足")

            previous_balance = number(row[previous_index])
            current_balance = number(row[current_index])

            if previous_balance is None or current_balance is None:
                raise ValueError("TWSE 融資餘額無法轉為數值")

            if previous_balance < 0 or current_balance < 0:
                raise ValueError("TWSE 融資餘額不可為負值")

            # 千元換算億元：除以 100,000。
            balance_yi = current_balance / 100000
            difference_yi = (
                current_balance - previous_balance
            ) / 100000

            return {
                # 為相容既有前端，沿用舊鍵名；
                # 實際單位是億元，而不是英文 billion。
                "balance_billion": round(balance_yi, 2),
                "daily_diff_billion": round(difference_yi, 2),
                "previous_balance_billion": round(
                    previous_balance / 100000, 2
                ),
                "unit": "億元",
                "est_maintenance_ratio": None,
                "data_date": data_date.isoformat(),
                "source": "TWSE 信用交易統計",
                "status": "ok",
                "note": (
                    "上市市場融資金額，原始單位為仟元；"
                    "今日餘額可能於後續調帳修正。"
                    "融資維持率無獨立資料，因此不估算。"
                ),
            }

    raise ValueError("找不到融資金額的前日餘額／今日餘額")


def fetch_twse_margin(session):
    errors = []

    # 不帶 date，向來源請求目前可提供的最新資料；
    # 實際資料日期以回應中的 date 為準。
    for endpoint in TWSE_ENDPOINTS:
        try:
            response = session.get(
                endpoint,
                params={
                    "response": "json",
                    "selectType": "ALL",
                },
                headers={
                    "Accept": "application/json",
                    "Referer": "https://www.twse.com.tw/",
                },
                timeout=(10, 25),
            )

            response.raise_for_status()

            try:
                payload = response.json()
            except ValueError as error:
                content_type = response.headers.get(
                    "Content-Type", "未提供"
                )

                raise ValueError(
                    "TWSE 未回傳有效 JSON；"
                    f"HTTP={response.status_code}，"
                    f"Content-Type={content_type}"
                ) from error

            result = parse_twse_payload(payload)
            result["source_url"] = response.url

            return result

        except (requests.RequestException, ValueError) as error:
            errors.append(f"{endpoint}: {error}")
            time.sleep(1)

    return {
        "balance_billion": None,
        "daily_diff_billion": None,
        "est_maintenance_ratio": None,
        "data_date": None,
        "source": "TWSE 信用交易統計",
        "status": "unavailable",
        "note": "；".join(errors),
    }


def fetch_twse_margin_with_fallback(session, previous_margin):
    attempted_at = taipei_now()

    try:
        current = fetch_twse_margin(session)
    except Exception as error:
        current = {
            "balance_billion": None,
            "daily_diff_billion": None,
            "est_maintenance_ratio": None,
            "data_date": None,
            "status": "unavailable",
            "note": str(error),
        }

    result = select_snapshot(
        current=current,
        previous=previous_margin,
        required_fields=["balance_billion"],
        attempted_at=attempted_at,
    )

    log(
        f"TWSE: status={result.get('status')}, "
        f"data_date={result.get('data_date')}, "
        f"balance={result.get('balance_billion')}"
    )

    if result.get("last_error"):
        log(f"TWSE 更新訊息：{result['last_error']}")

    return result


def fetch_treasury_year(session, year):
    url = f"{TREASURY_BASE_URL}/{year}/all"

    response = session.get(
        url,
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

    required = ["Date", "3 Mo", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]

    missing = [
        column for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"Treasury CSV 缺少欄位：{missing}")

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    for column in required[1:]:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    frame = frame.dropna(subset=required)

    for column in required[1:]:
        frame = frame[
            frame[column].map(
                lambda value: math.isfinite(float(value))
            )
        ]

    if frame.empty:
        raise ValueError(f"{year} 年 Treasury CSV 沒有有效資料")

    return frame


def valid_yields(snapshot):
    if not isinstance(snapshot, dict):
        return False

    if parse_date(snapshot.get("data_date")) is None:
        return False

    if snapshot.get("status") in {
        "unavailable", "error", "failed"
    }:
        return False

    values = snapshot.get("current")

    return (
        snapshot.get("maturities") ==
        ["3M", "2Y", "5Y", "10Y", "30Y"]
        and isinstance(values, list)
        and len(values) == 5
        and all(number(value) is not None for value in values)
    )


def fetch_treasury_yields(session, previous):
    attempted_at = taipei_now()

    try:
        year = dt.datetime.now(TAIPEI).year
        frames = []
        errors = []

        try:
            frames.append(fetch_treasury_year(session, year))
        except Exception as error:
            errors.append(str(error))

        if frames:
            latest_date = frames[0]["Date"].max()
            target_date = latest_date - pd.DateOffset(months=1)

            need_previous_year = (
                frames[0]["Date"].min() > target_date
            )
        else:
            need_previous_year = True

        if need_previous_year:
            try:
                frames.append(
                    fetch_treasury_year(session, year - 1)
                )
            except Exception as error:
                errors.append(str(error))

        if not frames:
            raise ValueError("；".join(errors))

        frame = pd.concat(frames, ignore_index=True)
        frame = frame.sort_values("Date")
        frame = frame.drop_duplicates("Date", keep="last")

        latest = frame.iloc[-1]
        target_date = latest["Date"] - pd.DateOffset(months=1)

        comparison_rows = frame[
            frame["Date"] <= target_date
        ]

        columns = ["3 Mo", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]

        current_values = [
            round(float(latest[column]), 2)
            for column in columns
        ]

        if comparison_rows.empty:
            comparison_values = []
            comparison_date = None
        else:
            comparison = comparison_rows.iloc[-1]

            comparison_values = [
                round(float(comparison[column]), 2)
                for column in columns
            ]

            comparison_date = comparison["Date"].strftime(
                "%Y-%m-%d"
            )

        current = {
            "maturities": ["3M", "2Y", "5Y", "10Y", "30Y"],
            "current": current_values,
            "one_month_ago": comparison_values,
            "spread_10y_2y": round(
                current_values[3] - current_values[1], 2
            ),
            "data_date": latest["Date"].strftime("%Y-%m-%d"),
            "comparison_date": comparison_date,
            "source": "U.S. Department of the Treasury",
            "status": "ok",
            "is_cached": False,
            "last_attempt_at": attempted_at,
            "last_success_at": attempted_at,
            "last_error": None,
            "note": (
                "殖利率單位為%；10Y−2Y 利差單位為百分點。"
                "比較期取資料日往前一個月當日或之前最近有效資料。"
            ),
        }

        if errors:
            current["note"] += " 部分年度請求失敗：" + "；".join(
                errors
            )

        if valid_yields(previous):
            old_date = parse_date(previous["data_date"])
            new_date = parse_date(current["data_date"])

            if new_date < old_date:
                raise ValueError(
                    f"來源回傳較舊資料日 {current['data_date']}"
                )

        log(
            "Treasury: status=ok, "
            f"data_date={current['data_date']}"
        )

        return current

    except Exception as error:
        log(f"Treasury 更新失敗：{error}")

        if valid_yields(previous):
            result = {
                **previous,
                "status": "cached",
                "is_cached": True,
                "last_attempt_at": attempted_at,
                "last_error": str(error),
            }

            result.setdefault("last_success_at", None)
            return result

        return {
            "maturities": ["3M", "2Y", "5Y", "10Y", "30Y"],
            "current": [],
            "one_month_ago": [],
            "spread_10y_2y": None,
            "data_date": None,
            "comparison_date": None,
            "source": "U.S. Department of the Treasury",
            "status": "unavailable",
            "is_cached": False,
            "last_attempt_at": attempted_at,
            "last_success_at": None,
            "last_error": str(error),
            "note": f"美債資料取得失敗：{error}",
        }


def empty_research_sections():
    """
    不沿用先前寫死的機率、短倉或投行觀點。
    待接入有來源、有日期的資料後，再標示 verified。
    """
    return {
        "fedwatch_dec": {
            "data_status": "unavailable",
            "meeting_date": None,
            "scenarios": [],
            "probabilities": [],
            "implied_rate_change_bps": None,
            "source": None,
            "data_date": None,
            "note": "尚未接入可驗證的利率機率來源。",
        },
        "short_interest_top_tech": {
            "data_status": "unavailable",
            "items": [],
            "source": None,
            "data_date": None,
            "note": "尚未接入可驗證的未平倉空單來源。",
        },
        "analyst_quant_report": {
            "data_status": "unavailable",
            "items": [],
            "source": None,
            "data_date": None,
            "note": "尚未接入可查核的機構研究報告。",
        },
    }


def main():
    log("開始更新市場資料")

    previous = load_previous_data()

    performance, history = fetch_indices_and_stocks(previous)

    with build_session() as session:
        margin = fetch_twse_margin_with_fallback(
            session,
            previous.get("tw_margin", {}),
        )

        treasury = fetch_treasury_yields(
            session,
            previous.get("treasury_yields", {}),
        )

    output = {
        "schema_version": 2,
        "last_updated": taipei_now(),
        "market_performance": performance,
        "normalized_history": history,
        "tw_margin": margin,
        "treasury_yields": treasury,
        **empty_research_sections(),
    }

    save_output(output)

    log(
        "data.json 已安全寫入；"
        f"TWSE={margin.get('status')}；"
        f"Treasury={treasury.get('status')}"
    )


if __name__ == "__main__":
    main()
