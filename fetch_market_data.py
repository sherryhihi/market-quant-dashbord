import json
import datetime
import requests
import yfinance as yf
import pandas as pd

STATIC_DATA_STATUS = "static_sample_not_realtime"


def fetch_indices_and_stocks():
    tickers = {
        "TWII": "^TWII",
        "TSMC": "2330.TW",
        "SOX": "^SOX",
        "NVDA": "NVDA",
        "IXIC": "^IXIC",
    }

    performance = {}
    history_series = {}

    for name, symbol in tickers.items():
        try:
            df = yf.Ticker(symbol).history(period="1mo", auto_adjust=False)

            if len(df) < 2:
                continue

            closes = df["Close"].dropna()

            if len(closes) < 2:
                continue

            latest_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2])
            month_start = float(closes.iloc[0])

            daily_chg = (latest_close / prev_close - 1) * 100
            month_chg = (latest_close / month_start - 1) * 100

            performance[name] = {
                "price": round(latest_close, 2),
                "daily_pct": round(daily_chg, 2),
                "month_pct": round(month_chg, 2),
            }

            normalized = closes / month_start * 100

            history_series[name] = {
                "dates": [d.strftime("%m-%d") for d in closes.index],
                "values": [round(float(v), 2) for v in normalized],
            }

        except Exception as e:
            print(f"Error fetching {name}: {e}")

    return performance, history_series


def get_recent_trading_date():
    today = datetime.date.today()

    for _ in range(10):
        if today.weekday() < 5:
            return today.strftime("%Y%m%d")
        today -= datetime.timedelta(days=1)

    return today.strftime("%Y%m%d")


def fetch_twse_margin():
    margin_info = {
        "balance_billion": None,
        "daily_diff_billion": None,
        "est_maintenance_ratio": None,
        "data_date": None,
        "note": "TWSE 信用交易統計；單位為億元",
        "status": "unavailable",
    }

    trade_date = get_recent_trading_date()
    url = (
        "https://www.twse.com.tw/rwd/zh/margin/mi_margn"
        f"?response=json&date={trade_date}&selectType=ALL"
    )

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.twse.com.tw/zh/page/margin/margin.html",
            },
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("retCode") != "0001":
            margin_info["note"] = f"TWSE API 回應異常：{payload.get('retMsg')}"
            return margin_info

        tables = payload.get("tables", [])
        if not tables:
            return margin_info

        fields = tables[0].get("fields", [])
        rows = tables[0].get("data", [])

        if not fields or not rows:
            return margin_info

        # TWSE 欄位可能因版本調整，先以欄位名稱定位
        field_index = {name.strip(): i for i, name in enumerate(fields)}

        financing_balance_idx = field_index.get("融資金額")
        financing_today_idx = field_index.get("融資今日餘額")
        financing_yesterday_idx = field_index.get("融資前日餘額")

        if financing_balance_idx is None and (
            financing_today_idx is None or financing_yesterday_idx is None
        ):
            margin_info["note"] = "TWSE 欄位結構已變更，請重新對應欄位名稱"
            return margin_info

        # 通常最後一列為「總計」
        row = rows[-1]

        def to_billion(value):
            if value in (None, "", "-"):
                return None
            return round(float(str(value).replace(",", "")) / 100000, 2)

        if financing_today_idx is not None and financing_yesterday_idx is not None:
            current = to_billion(row[financing_today_idx])
            previous = to_billion(row[financing_yesterday_idx])

            if current is not None:
                margin_info["balance_billion"] = current

            if current is not None and previous is not None:
                margin_info["daily_diff_billion"] = round(current - previous, 2)

        elif financing_balance_idx is not None:
            margin_info["balance_billion"] = to_billion(row[financing_balance_idx])

        margin_info["data_date"] = trade_date
        margin_info["status"] = "ok" if margin_info["balance_billion"] else "partial"

    except Exception as e:
        margin_info["note"] = f"TWSE 資料抓取失敗：{e}"

    return margin_info


def fetch_treasury_yields():
    yields_data = {
        "maturities": ["3M", "2Y", "5Y", "10Y", "30Y"],
        "current": [],
        "one_month_ago": [],
        "spread_10y_2y": None,
        "data_date": None,
        "status": "unavailable",
    }

    year = datetime.datetime.now().year
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/daily-treasury-rates.csv/"
        f"{year}/all?type=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}&page&_format=csv"
    )

    try:
        df = pd.read_csv(url)

        date_col = "Date"
        if date_col not in df.columns:
            yields_data["note"] = "Treasury CSV 欄位格式變更"
            return yields_data

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        latest = df.iloc[-1]
        one_month_ago = df.iloc[-22] if len(df) > 22 else df.iloc[0]

        column_map = {
            "3M": "3 Mo",
            "2Y": "2 Yr",
            "5Y": "5 Yr",
            "10Y": "10 Yr",
            "30Y": "30 Yr",
        }

        current = []
        previous = []

        for maturity in yields_data["maturities"]:
            current.append(float(latest[column_map[maturity]]))
            previous.append(float(one_month_ago[column_map[maturity]]))

        yields_data["current"] = [round(v, 2) for v in current]
        yields_data["one_month_ago"] = [round(v, 2) for v in previous]
        yields_data["spread_10y_2y"] = round(current[3] - current[1], 2)
        yields_data["data_date"] = latest[date_col].strftime("%Y-%m-%d")
        yields_data["status"] = "ok"

    except Exception as e:
        yields_data["note"] = f"Treasury 資料抓取失敗：{e}"

    return yields_data


def main():
    perf, hist = fetch_indices_and_stocks()
    margin = fetch_twse_margin()
    yields = fetch_treasury_yields()

    output = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "market_performance": perf,
        "normalized_history": hist,
        "tw_margin": margin,
        "treasury_yields": yields,
        "fedwatch_dec": {
            "data_status": STATIC_DATA_STATUS,
            "meeting_date": "12月 FOMC 利率決議",
            "scenarios": [
                "降息 1 碼 (25bps)",
                "維持利率不變",
                "升息 1 碼 (25bps)",
            ],
            "probabilities": [68.5, 31.5, 0.0],
            "implied_rate_change_bps": -17.1,
        },
        "short_interest_top_tech": {
            "data_status": STATIC_DATA_STATUS,
            "items": [
                {"ticker": "NVDA", "short_float_pct": 1.15, "days_to_cover": 0.9, "trend": "持平"},
                {"ticker": "TSLA", "short_float_pct": 3.42, "days_to_cover": 1.4, "trend": "微增"},
                {"ticker": "AMD", "short_float_pct": 2.85, "days_to_cover": 1.8, "trend": "下降"},
                {"ticker": "SMCI", "short_float_pct": 14.60, "days_to_cover": 2.6, "trend": "顯著上升"},
            ],
        },
        "analyst_quant_report": {
            "data_status": STATIC_DATA_STATUS,
            "items": [
                {
                    "institution": "Goldman Sachs (高盛)",
                    "tech_valuation_view": "偏高但具獲利支撐",
                    "target_forward_pe": "26x - 28x",
                    "defensive_allocation_pct": 30,
                    "strategy_summary": "減碼高本益比純軟體股，轉向具現金流護城河的AI基建與公用事業防禦組合。",
                },
                {
                    "institution": "Morgan Stanley (摩根士丹利)",
                    "tech_valuation_view": "面臨估值倍數修正壓力",
                    "target_forward_pe": "23x - 25x",
                    "defensive_allocation_pct": 40,
                    "strategy_summary": "建議提高醫療保健與必須消費權重，利用選擇權領口策略（Collar）對沖財報季波動。",
                },
                {
                    "institution": "Bank of America (美國銀行)",
                    "tech_valuation_view": "中性偏謹慎（泡沫警戒）",
                    "target_forward_pe": "24x - 26x",
                    "defensive_allocation_pct": 35,
                    "strategy_summary": "科技巨頭資本支出回報率受檢視，建議增持短天期美債與高股息價值股。",
                },
            ],
        },
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Successfully generated data.json")


if __name__ == "__main__":
    main()
