import json
import math
import os
import time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd

from scan import (
    MAIN_BOARD_PREFIXES,
    get_realtime_data,
    get_industry_heat_table,
    industry_confirmation,
    load_industry_cache,
    match_stock_industry,
    safe_float,
)

OUTPUT_FILE = "premarket_latest.json"
HISTORY_DIR = "premarket_history"
KLINE_POOL = 30
INDUSTRY_POOL = 20
NEWS_POOL = 12
NEWS_LOOKBACK_HOURS = 84
BEIJING = ZoneInfo("Asia/Shanghai")

POSITIVE_NEWS = {
    "预增": 2.4,
    "扭亏": 2.8,
    "中标": 2.0,
    "重大合同": 2.0,
    "订单": 1.2,
    "回购": 1.5,
    "增持": 1.5,
    "获批": 1.8,
    "突破": 1.2,
    "涨价": 1.0,
    "扩产": 1.0,
    "并购": 1.2,
    "收购": 1.0,
    "净利润增长": 1.5,
    "同比增长": 1.0,
}

NEGATIVE_NEWS = {
    "立案调查": -5.0,
    "退市风险": -5.0,
    "重大违法": -5.0,
    "预亏": -3.5,
    "首亏": -3.5,
    "减持": -2.0,
    "处罚": -2.5,
    "问询": -1.5,
    "亏损": -2.0,
    "下滑": -1.2,
    "终止": -1.8,
    "停产": -2.2,
    "诉讼": -1.8,
    "冻结": -2.0,
}

HARD_RISK_WORDS = ["立案调查", "退市风险", "重大违法", "预亏", "首亏", "停产"]


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_code(value):
    text = str(value).strip().lower()
    if text.startswith(("sh", "sz", "bj")):
        text = text[2:]
    return text.zfill(6)


def is_main_board(code):
    return str(code).startswith(MAIN_BOARD_PREFIXES)


def is_trade_day(date_obj):
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(cal["trade_date"], errors="coerce").dt.date.dropna())
        return date_obj in dates
    except Exception:
        return date_obj.weekday() < 5


def resolve_trade_dates(now):
    today = now.date()
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = sorted(pd.to_datetime(cal["trade_date"], errors="coerce").dt.date.dropna().unique())
        dates = [x for x in dates if hasattr(x, "year")]

        if today in dates and now.time() < dtime(15, 30):
            target = today
        else:
            future = [d for d in dates if d > today]
            target = future[0] if future else today

        previous = [d for d in dates if d < target]
        prev_trade = previous[-1] if previous else target
        return target, prev_trade
    except Exception:
        target = today
        if now.weekday() >= 5 or now.time() >= dtime(15, 30):
            target = today + timedelta(days=1)
            while target.weekday() >= 5:
                target += timedelta(days=1)
        prev_trade = target - timedelta(days=1)
        while prev_trade.weekday() >= 5:
            prev_trade -= timedelta(days=1)
        return target, prev_trade


def filter_universe(df):
    result = df.copy()
    result["code"] = result["code"].map(normalize_code)
    result = result[result["code"].map(is_main_board)]
    bad = result["name"].astype(str).str.upper().str.contains(r"ST|\*ST|退市", regex=True, na=False)
    result = result[~bad]
    result = result[(pd.to_numeric(result["price"], errors="coerce") >= 3) & (pd.to_numeric(result["price"], errors="coerce") <= 100)]
    return result


def target_score(x, target, width, max_score, neutral=0.45):
    x = safe_float(x)
    if x is None:
        return max_score * neutral
    return clamp(max_score * (1 - abs(x - target) / width), 0, max_score)


def coarse_score(row):
    d5 = safe_float(row.get("zdf_d5"))
    d10 = safe_float(row.get("zdf_d10"))
    d20 = safe_float(row.get("zdf_d20"))
    d60 = safe_float(row.get("zdf_d60"))
    amount = safe_float(row.get("amount"), 0) or 0
    turnover = safe_float(row.get("turnover_rate"))
    pe = safe_float(row.get("pe"))

    score = 0.0
    score += target_score(d5, 3.0, 12.0, 6.0)
    score += target_score(d10, 5.0, 20.0, 5.0)
    score += target_score(d20, 8.0, 28.0, 6.0)
    score += target_score(d60, 12.0, 55.0, 4.0)
    score += target_score(turnover, 4.0, 10.0, 3.0)

    if amount > 0:
        score += clamp((math.log10(max(amount, 1)) - 7) / 2.5 * 4, 0, 4)
    else:
        score += 1.5

    if pe is None:
        score += 1.5
    elif 5 <= pe <= 45:
        score += 2.0
    elif pe > 0:
        score += 1.0

    if d20 is not None and d20 > 20:
        score -= 4
    if d20 is not None and d20 > 30:
        score -= 5
    if d60 is not None and d60 > 40:
        score -= 4
    if d5 is not None and d5 > 10:
        score -= 4

    return round(clamp(score, 0, 30), 2)


def normalize_kline_frame(df):
    """
    把不同历史行情接口统一为中文列名，供后续日K评分使用。
    """
    if df is None or df.empty:
        raise RuntimeError("日K为空")

    out = df.copy()

    rename_map = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    }
    out.rename(columns=rename_map, inplace=True)

    required = ["收盘", "最高", "最低", "成交量"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError("日K缺少字段:" + ",".join(missing))

    return out


def fetch_kline(code, now):
    """
    V1.1:
    1. 腾讯历史日K优先，与当前实时行情数据源保持一致；
    2. 新浪作为备用；
    3. 不再使用 GitHub Actions 中连续 ConnectionError 的东方财富历史日K接口。
    """
    start = (now.date() - timedelta(days=200)).strftime("%Y%m%d")
    end = now.date().strftime("%Y%m%d")

    errors = []

    # ---------- 腾讯优先 ----------
    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=code,
            start_date=start,
            end_date=end,
            adjust="qfq",
            timeout=12,
        )
        out = normalize_kline_frame(df)
        out.attrs["kline_source"] = "腾讯历史日K"
        return out
    except Exception as e:
        errors.append(f"腾讯:{type(e).__name__}")
        time.sleep(0.15)

    # ---------- 新浪备用 ----------
    market = "sh" if code.startswith(("600", "601", "603", "605")) else "sz"
    try:
        df = ak.stock_zh_a_daily(
            symbol=f"{market}{code}",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        out = normalize_kline_frame(df)
        out.attrs["kline_source"] = "新浪历史日K"
        return out
    except Exception as e:
        errors.append(f"新浪:{type(e).__name__}")

    raise RuntimeError("历史日K双源失败 | " + " | ".join(errors))


def pct_change_from(close, n):
    if len(close) <= n:
        return None
    base = safe_float(close.iloc[-n - 1])
    last = safe_float(close.iloc[-1])
    if base is None or last is None or base == 0:
        return None
    return (last / base - 1) * 100


def kline_score(df):
    data = df.copy()
    for col in ["收盘", "最高", "最低", "成交量"]:
        if col not in data.columns:
            raise RuntimeError(f"日K缺少字段:{col}")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["收盘"]).reset_index(drop=True)
    if len(data) < 25:
        raise RuntimeError("日K样本不足25天")

    close = data["收盘"]
    high = data["最高"]
    volume = data["成交量"]
    last = float(close.iloc[-1])

    def ma(n):
        return float(close.tail(min(n, len(close))).mean())

    ma5, ma10, ma20, ma60 = ma(5), ma(10), ma(20), ma(60)
    ret5 = pct_change_from(close, 5)
    ret10 = pct_change_from(close, 10)
    ret20 = pct_change_from(close, 20)
    ret60 = pct_change_from(close, 60)

    high20 = float(high.tail(min(20, len(high))).max())
    high60 = float(high.tail(min(60, len(high))).max())
    dist20 = (last / high20 - 1) * 100 if high20 > 0 else None
    dist60 = (last / high60 - 1) * 100 if high60 > 0 else None

    prev20_volume = volume.iloc[-21:-1] if len(volume) >= 21 else volume.iloc[:-1]
    avg20_volume = safe_float(prev20_volume.mean())
    vol_ratio = (safe_float(volume.iloc[-1], 0) or 0) / avg20_volume if avg20_volume and avg20_volume > 0 else None

    trend = 0.0
    trend += 2.0 if last > ma5 else 0
    trend += 2.0 if last > ma10 else 0
    trend += 3.0 if last > ma20 else 0
    trend += 1.0 if last > ma60 else 0
    trend += 2.0 if ma5 > ma10 else 0
    trend += 1.0 if ma10 > ma20 else 0
    trend += 1.0 if ma20 > ma60 else 0

    if dist20 is None:
        position = 6.0
    elif -12 <= dist20 <= -4:
        position = 12.0
    elif -4 < dist20 <= -1:
        position = 10.0
    elif -18 <= dist20 < -12:
        position = 8.0
    elif dist20 > -1:
        position = 6.0
    else:
        position = 5.0

    momentum = 0.0
    momentum += target_score(ret5, 2.5, 9.0, 4.0, neutral=0.5)
    momentum += target_score(ret20, 7.0, 22.0, 4.0, neutral=0.5)
    momentum += target_score(ret60, 12.0, 45.0, 2.0, neutral=0.5)

    if vol_ratio is None:
        volume_score = 3.0
    elif 0.8 <= vol_ratio <= 1.8:
        volume_score = 6.0
    elif 0.55 <= vol_ratio < 0.8 or 1.8 < vol_ratio <= 2.5:
        volume_score = 4.5
    elif vol_ratio > 3.5:
        volume_score = 1.5
    else:
        volume_score = 3.0

    penalty = 0.0
    flags = []
    if ret5 is not None and ret5 > 9:
        penalty += 3
        flags.append("5日涨幅偏高")
    if ret20 is not None and ret20 > 18:
        penalty += 5
        flags.append("20日位置偏高")
    if ret20 is not None and ret20 > 28:
        penalty += 4
        flags.append("20日明显过热")
    if ret60 is not None and ret60 > 35:
        penalty += 4
        flags.append("60日涨幅偏高")
    if dist20 is not None and dist20 > -0.8 and ret20 is not None and ret20 > 10:
        penalty += 4
        flags.append("接近20日新高且已有涨幅")
    if vol_ratio is not None and vol_ratio > 3.5:
        penalty += 2
        flags.append("成交量异常放大")

    score = clamp(trend + position + momentum + volume_score - penalty, 0, 40)

    return {
        "score": round(score, 2),
        "close": round(last, 3),
        "ma5": round(ma5, 3),
        "ma10": round(ma10, 3),
        "ma20": round(ma20, 3),
        "ma60": round(ma60, 3),
        "ret5": round(ret5, 2) if ret5 is not None else None,
        "ret10": round(ret10, 2) if ret10 is not None else None,
        "ret20": round(ret20, 2) if ret20 is not None else None,
        "ret60": round(ret60, 2) if ret60 is not None else None,
        "distance_20d_high": round(dist20, 2) if dist20 is not None else None,
        "distance_60d_high": round(dist60, 2) if dist60 is not None else None,
        "volume_ratio_20d": round(vol_ratio, 2) if vol_ratio is not None else None,
        "risk_flags": flags,
    }


def report_periods(now):
    year = now.year
    candidates = [
        f"{year}0930",
        f"{year}0630",
        f"{year}0331",
        f"{year - 1}1231",
        f"{year - 1}0930",
    ]
    today_str = now.strftime("%Y%m%d")
    return [x for x in candidates if x <= today_str]


def load_fundamental_tables(now):
    periods = report_periods(now)
    reports = []
    used_periods = []

    for period in periods[:3]:
        try:
            print(f"获取业绩报表: {period}")
            df = ak.stock_yjbb_em(date=period)
            if df is not None and not df.empty:
                df = df.copy()
                df["股票代码"] = df["股票代码"].map(normalize_code)
                df["report_period"] = period
                reports.append(df)
                used_periods.append(period)
            if reports and len(pd.concat(reports, ignore_index=True)["股票代码"].unique()) > 3500:
                break
        except Exception as e:
            print("业绩报表失败:", period, type(e).__name__)

    report_map = {}
    if reports:
        merged = pd.concat(reports, ignore_index=True)
        merged = merged.sort_values("report_period", ascending=False).drop_duplicates("股票代码", keep="first")
        report_map = {str(row["股票代码"]): row.to_dict() for _, row in merged.iterrows()}

    forecast_map = {}
    if periods:
        for period in periods[:2]:
            try:
                print(f"获取业绩预告: {period}")
                fdf = ak.stock_yjyg_em(date=period)
                if fdf is not None and not fdf.empty:
                    fdf = fdf.copy()
                    fdf["股票代码"] = fdf["股票代码"].map(normalize_code)
                    if "公告日期" in fdf.columns:
                        fdf = fdf.sort_values("公告日期", ascending=False)
                    fdf = fdf.drop_duplicates("股票代码", keep="first")
                    forecast_map = {str(row["股票代码"]): row.to_dict() for _, row in fdf.iterrows()}
                    break
            except Exception as e:
                print("业绩预告失败:", period, type(e).__name__)

    return report_map, forecast_map, used_periods


def fundamental_score(code, report_map, forecast_map):
    report = report_map.get(code)
    forecast = forecast_map.get(code)

    if report is None:
        rev_yoy = profit_yoy = roe = None
        report_period = None
        score = 10.5
    else:
        rev_yoy = safe_float(report.get("营业总收入-同比增长"))
        if rev_yoy is None:
            rev_yoy = safe_float(report.get("营业收入-同比增长"))
        profit_yoy = safe_float(report.get("净利润-同比增长"))
        roe = safe_float(report.get("净资产收益率"))
        report_period = report.get("report_period")

        if rev_yoy is None:
            rev_score = 3.0
        elif rev_yoy < -20:
            rev_score = 0.5
        elif rev_yoy < 0:
            rev_score = 2.0
        elif rev_yoy < 15:
            rev_score = 4.0
        elif rev_yoy < 30:
            rev_score = 5.0
        else:
            rev_score = 6.0

        if profit_yoy is None:
            profit_score = 5.0
        elif profit_yoy < -50:
            profit_score = 0.5
        elif profit_yoy < 0:
            profit_score = 2.0
        elif profit_yoy < 20:
            profit_score = 6.0
        elif profit_yoy < 50:
            profit_score = 8.0
        else:
            profit_score = 10.0

        if roe is None:
            roe_score = 2.5
        elif roe <= 0:
            roe_score = 0.5
        elif roe < 5:
            roe_score = 3.0
        elif roe < 10:
            roe_score = 4.0
        else:
            roe_score = 5.0

        score = rev_score + profit_score + roe_score

    forecast_type = None
    forecast_change = None
    if forecast is not None:
        forecast_type = str(forecast.get("预告类型", ""))
        forecast_change = safe_float(forecast.get("业绩变动幅度"))
        positive_type = any(x in forecast_type for x in ["预增", "扭亏", "略增", "续盈"])
        negative_type = any(x in forecast_type for x in ["预亏", "首亏", "预减", "略减", "续亏"])
        if positive_type:
            score += 2.0
            if forecast_change is not None and forecast_change >= 30:
                score += 2.0
        elif negative_type:
            score -= 4.0

    return {
        "score": round(clamp(score, 0, 25), 2),
        "report_period": report_period,
        "revenue_yoy": round(rev_yoy, 2) if rev_yoy is not None else None,
        "profit_yoy": round(profit_yoy, 2) if profit_yoy is not None else None,
        "roe": round(roe, 2) if roe is not None else None,
        "forecast_type": forecast_type or None,
        "forecast_change": round(forecast_change, 2) if forecast_change is not None else None,
    }


def parse_news_time(value):
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return None
        py = dt.to_pydatetime()
        if py.tzinfo is None:
            py = py.replace(tzinfo=BEIJING)
        else:
            py = py.astimezone(BEIJING)
        return py
    except Exception:
        return None


def news_score(code, now):
    neutral = 12.5
    score = neutral
    headlines = []
    flags = []
    try:
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            return {"score": neutral, "headlines": [], "risk_flags": [], "news_count": 0}

        used = 0
        for _, row in df.iterrows():
            published = parse_news_time(row.get("发布时间"))
            if published is None:
                age_hours = 48
            else:
                age_hours = max(0, (now - published).total_seconds() / 3600)
            if age_hours > NEWS_LOOKBACK_HOURS:
                continue

            title = str(row.get("新闻标题", "")).strip()
            content = str(row.get("新闻内容", "")).strip()
            text = title + " " + content[:220]
            weight = 1.0 if age_hours <= 24 else 0.7 if age_hours <= 48 else 0.45

            delta = 0.0
            for word, points in POSITIVE_NEWS.items():
                if word in text:
                    delta += points * weight
            for word, points in NEGATIVE_NEWS.items():
                if word in text:
                    delta += points * weight
                    if word in HARD_RISK_WORDS:
                        flags.append(word)

            score += clamp(delta, -6.0, 4.0)
            if title and len(headlines) < 3:
                headlines.append({
                    "title": title,
                    "published_at": published.isoformat() if published else None,
                    "source": str(row.get("文章来源", "")) or None,
                    "url": str(row.get("新闻链接", "")) or None,
                })
            used += 1

        return {
            "score": round(clamp(score, 0, 25), 2),
            "headlines": headlines,
            "risk_flags": list(dict.fromkeys(flags)),
            "news_count": used,
        }
    except Exception as e:
        print(code, "新闻获取失败:", type(e).__name__)
        return {"score": neutral, "headlines": [], "risk_flags": [], "news_count": 0}


def industry_info_for_candidates(candidates):
    try:
        table = get_industry_heat_table()
    except Exception as e:
        print("行业热度失败:", type(e).__name__)
        for item in candidates:
            item["industry"] = None
            item["industry_score"] = 5.0
            item["sector_confirmation"] = "未知"
        return

    cache = load_industry_cache()
    for item in candidates:
        code = item["code"]
        try:
            industry, source_field, raw_class = match_stock_industry(code, table, cache)
            item["industry"] = industry
            item["industry_match_field"] = source_field
            item["industry_raw_class"] = raw_class
            if industry is None:
                item["industry_score"] = 5.0
                item["sector_confirmation"] = "未知"
                continue
            matched = table[table["industry"] == industry]
            if matched.empty:
                item["industry_score"] = 5.0
                item["sector_confirmation"] = "未知"
                continue
            row = matched.iloc[0]
            heat = safe_float(row.get("heat_score"), 5.0)
            item["industry_score"] = round(clamp(heat if heat is not None else 5.0, 0, 10), 2)
            item["sector_confirmation"] = industry_confirmation(heat)
            item["industry_change_now"] = safe_float(row.get("change_now"))
            item["industry_change_3d"] = safe_float(row.get("change_3d"))
            item["industry_net_now"] = safe_float(row.get("net_now"))
            item["industry_net_3d"] = safe_float(row.get("net_3d"))
        except Exception as e:
            print(code, "行业匹配失败:", type(e).__name__)
            item["industry"] = None
            item["industry_score"] = 5.0
            item["sector_confirmation"] = "未知"


def build_reason(item):
    reasons = []
    k = item.get("kline", {})
    f = item.get("fundamental", {})
    dist = k.get("distance_20d_high")
    ret20 = k.get("ret20")

    if dist is not None and -12 <= dist <= -1:
        reasons.append("日K未处于明显追高区")
    if k.get("close") and k.get("ma20") and k["close"] > k["ma20"]:
        reasons.append("收盘价站上20日均线")
    if ret20 is not None and 0 <= ret20 <= 15:
        reasons.append("20日涨幅温和")
    if f.get("profit_yoy") is not None and f["profit_yoy"] > 20:
        reasons.append(f"净利润同比+{f['profit_yoy']:.1f}%")
    if f.get("forecast_type") in ["预增", "扭亏", "略增", "续盈"]:
        reasons.append(f"业绩预告{f['forecast_type']}")
    if item.get("sector_confirmation") in ["强", "偏强"]:
        reasons.append(f"{item.get('industry') or '行业'}{item['sector_confirmation']}确认")
    if item.get("news", {}).get("score", 12.5) >= 16:
        reasons.append("近几日新闻催化偏正面")
    if not reasons:
        reasons.append("综合评分进入盘前候选")
    return reasons[:4]


def build_risk_flags(item):
    flags = []
    flags.extend(item.get("kline", {}).get("risk_flags", []))
    flags.extend(item.get("news", {}).get("risk_flags", []))
    if item.get("sector_confirmation") == "弱":
        flags.append("行业确认偏弱")
    if item.get("fundamental", {}).get("report_period") is None:
        flags.append("最新业绩数据缺失/未披露")
    return list(dict.fromkeys(flags))[:6]


def recommendation_label(score, risk_flags):
    hard = any(x in risk_flags for x in HARD_RISK_WORDS)
    if hard:
        return "回避"
    if score >= 80:
        return "重点候选"
    if score >= 72:
        return "候选"
    return "观察"


def save_payload(payload, target_date):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(HISTORY_DIR, f"{target_date}.json")
    if not os.path.exists(history_path) and payload.get("status") == "ok":
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        print("已冻结盘前首份结果:", history_path)
    elif os.path.exists(history_path):
        print("盘前历史已存在，不覆盖:", history_path)


def main():
    now = datetime.now(BEIJING)
    target_date, prev_trade = resolve_trade_dates(now)
    print("=" * 64)
    print("A股盘前量化 V1.1")
    print("运行时间:", now.isoformat())
    print("目标交易日:", target_date)
    print("参考最近交易日:", prev_trade)
    print("=" * 64)

    try:
        raw_df, source, quality = get_realtime_data()
        universe = filter_universe(raw_df)
        universe["coarse_score"] = universe.apply(coarse_score, axis=1)
        coarse = universe.sort_values(["coarse_score", "amount"], ascending=[False, False]).head(KLINE_POOL)
        print(f"主板候选池: {len(universe)} -> 日K精筛 {len(coarse)}")

        candidates = []
        for idx, (_, row) in enumerate(coarse.iterrows(), 1):
            code = normalize_code(row["code"])
            try:
                kdf = fetch_kline(code, now)
                kinfo = kline_score(kdf)
                kinfo["source"] = kdf.attrs.get("kline_source", "未知")
            except Exception as e:
                print(f"[{idx:02d}/{len(coarse)}] {code} 日K失败: {type(e).__name__}")
                continue
            candidates.append({
                "code": code,
                "name": str(row["name"]),
                "reference_price": safe_float(row.get("price")),
                "pe": safe_float(row.get("pe")),
                "pb": safe_float(row.get("pb")),
                "amount": safe_float(row.get("amount")),
                "coarse_score": safe_float(row.get("coarse_score"), 0.0),
                "kline": kinfo,
            })
            print(f"[{idx:02d}/{len(coarse)}] {code} {row['name']} K={kinfo['score']}")
            time.sleep(0.08)

        if len(candidates) < 8:
            raise RuntimeError("有效日K候选不足8只")

        report_map, forecast_map, used_periods = load_fundamental_tables(now)
        for item in candidates:
            item["fundamental"] = fundamental_score(item["code"], report_map, forecast_map)
            item["pre_industry_score"] = round(
                item["kline"]["score"] + item["fundamental"]["score"] + 12.5,
                2,
            )

        candidates.sort(key=lambda x: x["pre_industry_score"], reverse=True)
        industry_candidates = candidates[:INDUSTRY_POOL]
        industry_info_for_candidates(industry_candidates)

        for item in industry_candidates:
            item["pre_news_score"] = round(
                item["kline"]["score"]
                + item["fundamental"]["score"]
                + item.get("industry_score", 5.0)
                + 12.5,
                2,
            )

        industry_candidates.sort(key=lambda x: x["pre_news_score"], reverse=True)
        news_candidates = industry_candidates[:NEWS_POOL]

        for idx, item in enumerate(news_candidates, 1):
            print(f"新闻 [{idx:02d}/{len(news_candidates)}] {item['code']} {item['name']}")
            item["news"] = news_score(item["code"], now)
            item["total_score"] = round(
                item["kline"]["score"]
                + item["fundamental"]["score"]
                + item.get("industry_score", 5.0)
                + item["news"]["score"],
                2,
            )
            item["risk_flags"] = build_risk_flags(item)
            item["reasons"] = build_reason(item)
            item["recommendation"] = recommendation_label(item["total_score"], item["risk_flags"])
            item["auction_rule"] = "09:25竞价涨幅建议-1.5%~+2.5%；>+3%默认不追，<-2.5%需重新确认"
            item["shadow_rule"] = "09:35~09:40再用盘中Shadow确认；行业/个股明显转弱则取消"
            time.sleep(0.12)

        news_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        top10 = news_candidates[:10]
        for rank, item in enumerate(top10, 1):
            item["rank"] = rank

        payload = {
            "status": "ok",
            "strategy": "盘前量化 V1.1·日K40+业绩25+新闻25+行业10",
            "run_time": now.isoformat(),
            "target_trade_date": str(target_date),
            "reference_trade_date": str(prev_trade),
            "data_source": source,
            "data_quality": quality,
            "universe_count": int(len(universe)),
            "kline_pool_count": int(len(coarse)),
            "valid_kline_count": int(len(candidates)),
            "news_pool_count": int(len(news_candidates)),
            "fundamental_periods": used_periods,
            "top5": top10[:5],
            "top10": top10,
            "execution_note": "盘前榜单解决买什么，不等于开盘立即买；09:25竞价和09:35~09:40 Shadow必须二次确认，避免追高。",
        }
        save_payload(payload, target_date)

        print("\n盘前 Top5:")
        for item in top10[:5]:
            print(
                f"{item['rank']:02d}. {item['code']} {item['name']} "
                f"总分={item['total_score']} | K={item['kline']['score']} "
                f"业绩={item['fundamental']['score']} 新闻={item['news']['score']} "
                f"行业={item.get('industry_score', 5.0)} | {item['recommendation']}"
            )

    except Exception as e:
        print("盘前量化失败:", repr(e))
        payload = {
            "status": "error",
            "strategy": "盘前量化 V1.1·日K40+业绩25+新闻25+行业10",
            "run_time": now.isoformat(),
            "target_trade_date": str(target_date),
            "reference_trade_date": str(prev_trade),
            "message": str(e),
            "top5": [],
            "top10": [],
        }
        save_payload(payload, target_date)
        raise


if __name__ == "__main__":
    main()
