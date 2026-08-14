import json
import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd


OUTPUT_FILE = "latest.json"

# 只保留你可以买的沪深主板
MAIN_BOARD_PREFIXES = (
    "000", "001", "002", "003",
    "600", "601", "603", "605"
)


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_code(code):
    """
    新浪返回类似 sz002185 / sh600312
    转换成 002185 / 600312
    """
    code = str(code).strip().lower()

    if code.startswith(("sh", "sz", "bj")):
        code = code[2:]

    return code.zfill(6)


def get_realtime_data():
    """
    第一数据源：东方财富
    如果失败则自动切换新浪。
    """

    # ---------- 东方财富 ----------
    try:
        print("正在尝试东方财富实时行情...")
        df = ak.stock_zh_a_spot_em()

        if df is not None and not df.empty:
            print(f"东方财富成功：{len(df)} 条")

            result = pd.DataFrame({
                "code": df["代码"].astype(str).str.zfill(6),
                "name": df["名称"],
                "price": pd.to_numeric(df["最新价"], errors="coerce"),
                "change_pct": pd.to_numeric(df["涨跌幅"], errors="coerce"),
                "pre_close": pd.to_numeric(df["昨收"], errors="coerce"),
                "open": pd.to_numeric(df["今开"], errors="coerce"),
                "high": pd.to_numeric(df["最高"], errors="coerce"),
                "low": pd.to_numeric(df["最低"], errors="coerce"),
                "volume": pd.to_numeric(df["成交量"], errors="coerce"),
                "amount": pd.to_numeric(df["成交额"], errors="coerce"),
            })

            # 东财额外字段
            result["turnover_rate"] = pd.to_numeric(
                df.get("换手率"), errors="coerce"
            )
            result["volume_ratio"] = pd.to_numeric(
                df.get("量比"), errors="coerce"
            )
            result["pe"] = pd.to_numeric(
                df.get("市盈率-动态"), errors="coerce"
            )
            result["pb"] = pd.to_numeric(
                df.get("市净率"), errors="coerce"
            )

            return result, "东方财富A股实时行情", "full"

    except Exception as e:
        print("东方财富失败：", repr(e))

    # ---------- 新浪备用 ----------
    print("切换新浪A股实时行情...")

    start = time.time()
    df = ak.stock_zh_a_spot()

    if df is None or df.empty:
        raise RuntimeError("新浪行情返回空数据")

    elapsed = time.time() - start
    print(f"新浪成功：{len(df)} 条，耗时 {elapsed:.1f} 秒")

    result = pd.DataFrame({
        "code": df["代码"].map(normalize_code),
        "name": df["名称"],
        "price": pd.to_numeric(df["最新价"], errors="coerce"),
        "change_pct": pd.to_numeric(df["涨跌幅"], errors="coerce"),
        "pre_close": pd.to_numeric(df["昨收"], errors="coerce"),
        "open": pd.to_numeric(df["今开"], errors="coerce"),
        "high": pd.to_numeric(df["最高"], errors="coerce"),
        "low": pd.to_numeric(df["最低"], errors="coerce"),
        "volume": pd.to_numeric(df["成交量"], errors="coerce"),
        "amount": pd.to_numeric(df["成交额"], errors="coerce"),
    })

    # 新浪没有这些字段，保持为空，不当成0分
    result["turnover_rate"] = np.nan
    result["volume_ratio"] = np.nan
    result["pe"] = np.nan
    result["pb"] = np.nan

    return result, "新浪A股实时行情", "partial"


def filter_main_board(df):
    df = df.copy()

    # 代码范围
    df = df[
        df["code"].astype(str).str.startswith(MAIN_BOARD_PREFIXES)
    ]

    # 排除 ST / *ST / 退市
    bad_name = (
        df["name"]
        .astype(str)
        .str.upper()
        .str.contains(r"ST|\*ST|退市", regex=True, na=False)
    )

    df = df[~bad_name]

    # 排除无价格/停牌/异常值
    df = df[
        (df["price"] > 0)
        & (df["pre_close"] > 0)
        & (df["amount"].fillna(0) > 0)
    ]

    return df


def score_stock(row):
    """
    V1 实时量化模型
    核心原则：
    - 不追涨停
    - 不只选当天涨幅最大的
    - 偏好温和走强、成交活跃、波动不过度
    """

    price = safe_float(row["price"], 0)
    change = safe_float(row["change_pct"], 0)
    pre_close = safe_float(row["pre_close"], 0)
    open_price = safe_float(row["open"], pre_close)
    high = safe_float(row["high"], price)
    low = safe_float(row["low"], price)
    amount = safe_float(row["amount"], 0)

    score = 0.0

    # ------------------------------------------------
    # 1. 当日趋势：25分
    # ------------------------------------------------
    if -1.0 <= change < 0:
        trend_score = 10
    elif 0 <= change < 1:
        trend_score = 15
    elif 1 <= change <= 3:
        trend_score = 25
    elif 3 < change <= 5:
        trend_score = 21
    elif 5 < change <= 7:
        trend_score = 13
    elif change > 7:
        trend_score = 4
    else:
        trend_score = 3

    score += trend_score

    # ------------------------------------------------
    # 2. 流动性：20分
    # ------------------------------------------------
    # 成交额单位：元
    if amount >= 1_000_000_000:
        liquidity_score = 20
    elif amount >= 500_000_000:
        liquidity_score = 18
    elif amount >= 200_000_000:
        liquidity_score = 15
    elif amount >= 100_000_000:
        liquidity_score = 12
    elif amount >= 30_000_000:
        liquidity_score = 8
    else:
        liquidity_score = 3

    score += liquidity_score

    # ------------------------------------------------
    # 3. 日内强度：15分
    # ------------------------------------------------
    if high > low:
        position = (price - low) / (high - low)
        position = clamp(position, 0, 1)
    else:
        position = 0.5

    if 0.60 <= position <= 0.85:
        intraday_score = 15
    elif 0.45 <= position < 0.60:
        intraday_score = 11
    elif 0.85 < position <= 0.95:
        intraday_score = 10
    elif position > 0.95:
        intraday_score = 6
    else:
        intraday_score = 5

    score += intraday_score

    # ------------------------------------------------
    # 4. 振幅风险：15分
    # ------------------------------------------------
    if pre_close > 0:
        amplitude = (high - low) / pre_close * 100
    else:
        amplitude = 0

    if 1 <= amplitude <= 4:
        amplitude_score = 15
    elif 4 < amplitude <= 6:
        amplitude_score = 12
    elif amplitude < 1:
        amplitude_score = 8
    elif 6 < amplitude <= 8:
        amplitude_score = 7
    else:
        amplitude_score = 3

    score += amplitude_score

    # ------------------------------------------------
    # 5. 开盘后承接：10分
    # ------------------------------------------------
    if open_price > 0:
        vs_open = (price - open_price) / open_price * 100
    else:
        vs_open = 0

    if 0.2 <= vs_open <= 2.5:
        open_score = 10
    elif 0 <= vs_open < 0.2:
        open_score = 8
    elif -0.8 <= vs_open < 0:
        open_score = 6
    elif 2.5 < vs_open <= 4:
        open_score = 6
    else:
        open_score = 3

    score += open_score

    # ------------------------------------------------
    # 6. 防追高：15分
    # ------------------------------------------------
    if change <= 2:
        chase_score = 15
    elif change <= 4:
        chase_score = 12
    elif change <= 5.5:
        chase_score = 8
    elif change <= 7:
        chase_score = 4
    else:
        chase_score = 0

    score += chase_score

    return round(score, 2), round(amplitude, 2), round(position, 3)


def build_results(df):
    records = []

    for _, row in df.iterrows():
        score, amplitude, intraday_position = score_stock(row)

        item = {
            "code": str(row["code"]),
            "name": str(row["name"]),
            "price": safe_float(row["price"]),
            "change_pct": safe_float(row["change_pct"]),
            "open": safe_float(row["open"]),
            "high": safe_float(row["high"]),
            "low": safe_float(row["low"]),
            "pre_close": safe_float(row["pre_close"]),
            "volume": safe_float(row["volume"]),
            "amount": safe_float(row["amount"]),
            "turnover_rate": safe_float(row["turnover_rate"]),
            "volume_ratio": safe_float(row["volume_ratio"]),
            "pe": safe_float(row["pe"]),
            "pb": safe_float(row["pb"]),
            "amplitude": amplitude,
            "intraday_position": intraday_position,
            "score": score,
        }

        records.append(item)

    # 分数优先，成交额作为次排序
    records.sort(
        key=lambda x: (
            x["score"],
            x["amount"] if x["amount"] is not None else 0
        ),
        reverse=True
    )

    return records[:20]


def save_json(payload):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )


def main():
    beijing = ZoneInfo("Asia/Shanghai")
    now = datetime.now(beijing)

    print("=" * 60)
    print("A股量化扫描开始")
    print("北京时间：", now.isoformat())
    print("=" * 60)

    try:
        raw_df, data_source, data_quality = get_realtime_data()

        raw_count = len(raw_df)

        filtered_df = filter_main_board(raw_df)
        filtered_count = len(filtered_df)

        top20 = build_results(filtered_df)

        payload = {
            "status": "ok",
            "data_time": now.isoformat(),
            "calculated_at": datetime.now(beijing).isoformat(),
            "data_source": data_source,
            "data_quality": data_quality,
            "raw_stock_count": raw_count,
            "main_board_count": filtered_count,
            "result_count": len(top20),
            "strategy": "主板非ST・偏低位・避免追高 V1",
            "stocks": top20,
        }

        save_json(payload)

        print("")
        print("扫描成功")
        print("数据源：", data_source)
        print("原始数量：", raw_count)
        print("主板过滤后：", filtered_count)
        print("最终结果：", len(top20))

        print("")
        print("Top 10：")

        for i, stock in enumerate(top20[:10], 1):
            print(
                f"{i:02d}. "
                f"{stock['code']} "
                f"{stock['name']} "
                f"价格={stock['price']} "
                f"涨幅={stock['change_pct']}% "
                f"评分={stock['score']}"
            )

    except Exception as e:
        print("扫描失败：", repr(e))

        # 即使失败，也生成结构化 latest.json
        # 防止网页或 Actions 因文件不存在而完全失效
        payload = {
            "status": "error",
            "data_time": now.isoformat(),
            "calculated_at": datetime.now(beijing).isoformat(),
            "data_source": None,
            "data_quality": None,
            "message": str(e),
            "stocks": [],
        }

        save_json(payload)

        raise


if __name__ == "__main__":
    main()
