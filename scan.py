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
    第一数据源：腾讯
    第二数据源：新浪
    """

    # ---------- 腾讯优先 ----------
    try:
        print("正在尝试腾讯A股实时行情...")

        start = time.time()
        df = ak.stock_zh_a_spot_tx()
        elapsed = time.time() - start

        if df is not None and not df.empty:
            print(
                f"腾讯成功：{len(df)} 条，耗时 {elapsed:.1f} 秒"
            )

            price = pd.to_numeric(
                df["zxj"], errors="coerce"
            )
            change_pct = pd.to_numeric(
                df["zdf"], errors="coerce"
            )
            amplitude_tx = pd.to_numeric(
                df["zf"], errors="coerce"
            )

            # 根据最新价和涨跌幅反推昨收
            pre_close = price / (1 + change_pct / 100.0)

            # 腾讯全市场榜单没有直接提供今开/最高/最低
            # 这里先用中性代理值兼容现有 V1 模型
            half_range = pre_close * amplitude_tx / 200.0

            result = pd.DataFrame({
                "code": df["code"].map(normalize_code),
                "name": df["name"].astype(str),
                "price": price,
                "change_pct": change_pct,
                "pre_close": pre_close,
                "open": pre_close,
                "high": price + half_range,
                "low": (price - half_range).clip(lower=0),

                # 腾讯 volume 单位为手，转换为股
                "volume": pd.to_numeric(
                    df["volume"], errors="coerce"
                ) * 100,

                # turnover 转换为元
                "amount": pd.to_numeric(
                    df["turnover"], errors="coerce"
                ) * 10000,
            })

            if "hsl" in df.columns:
                result["turnover_rate"] = pd.to_numeric(
                    df["hsl"], errors="coerce"
                )
            elif "hs1" in df.columns:
                result["turnover_rate"] = pd.to_numeric(
                    df["hs1"], errors="coerce"
                )
            else:
                result["turnover_rate"] = np.nan

            result["volume_ratio"] = pd.to_numeric(
                df["lb"], errors="coerce"
            )

            result["pe"] = pd.to_numeric(
                df["pe_ttm"], errors="coerce"
            )

            result["pb"] = pd.to_numeric(
                df["pn"], errors="coerce"
            )
            result["speed"] = pd.to_numeric(
                df["speed"], errors="coerce"
            )

            result["zdf_d5"] = pd.to_numeric(
                df["zdf_d5"], errors="coerce"
            )

            result["zdf_d10"] = pd.to_numeric(
               df["zdf_d10"], errors="coerce"
            )

            result["zdf_d20"] = pd.to_numeric(
               df["zdf_d20"], errors="coerce"
            )

            result["zdf_d60"] = pd.to_numeric(
               df["zdf_d60"], errors="coerce"
            )

            result["amplitude_tx"] = pd.to_numeric(
              df["zf"], errors="coerce"
            )
            return (
                result,
                "腾讯A股实时行情",
                "partial"
            )

    except Exception as e:
        print("腾讯失败：", repr(e))

    # ---------- 新浪备用 ----------
    try:
        print("切换新浪A股实时行情...")

        start = time.time()
        df = ak.stock_zh_a_spot()
        elapsed = time.time() - start

        if df is None or df.empty:
            raise RuntimeError("新浪行情返回空数据")

        print(
            f"新浪成功：{len(df)} 条，耗时 {elapsed:.1f} 秒"
        )

        result = pd.DataFrame({
            "code": df["代码"].map(normalize_code),
            "name": df["名称"],
            "price": pd.to_numeric(
                df["最新价"], errors="coerce"
            ),
            "change_pct": pd.to_numeric(
                df["涨跌幅"], errors="coerce"
            ),
            "pre_close": pd.to_numeric(
                df["昨收"], errors="coerce"
            ),
            "open": pd.to_numeric(
                df["今开"], errors="coerce"
            ),
            "high": pd.to_numeric(
                df["最高"], errors="coerce"
            ),
            "low": pd.to_numeric(
                df["最低"], errors="coerce"
            ),
            "volume": pd.to_numeric(
                df["成交量"], errors="coerce"
            ),
            "amount": pd.to_numeric(
                df["成交额"], errors="coerce"
            ),
        })

        result["turnover_rate"] = np.nan
        result["volume_ratio"] = np.nan
        result["pe"] = np.nan
        result["pb"] = np.nan

        return (
            result,
            "新浪A股实时行情",
            "partial"
        )

    except Exception as e:
        print("新浪失败：", repr(e))

    raise RuntimeError("腾讯和新浪实时行情均获取失败")


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
    V2 量化评分
    核心思路：
    1. 偏好低位温和启动
    2. 避免连续快速上涨后追高
    3. 看 5/10/20/60 日趋势
    4. 看量比、换手、流动性
    5. 控制振幅和涨速风险

    满分 100
    """

    def value(name, default=None):
        return safe_float(row.get(name), default)

    def target_score(x, target, width, max_score, neutral=0.45):
        """
        越接近理想值，得分越高。
        缺失数据给予中性分，不直接判0。
        """
        if x is None:
            return max_score * neutral

        distance = abs(x - target)

        score = max_score * (
            1 - distance / width
        )

        return clamp(score, 0, max_score)

    # =========================
    # 数据读取
    # =========================

    change = value("change_pct", 0)

    d5 = value("zdf_d5")
    d10 = value("zdf_d10")
    d20 = value("zdf_d20")
    d60 = value("zdf_d60")

    volume_ratio = value("volume_ratio")
    turnover_rate = value("turnover_rate")
    pe = value("pe")

    speed = value("speed")

    amount = value("amount", 0)

    amplitude = value("amplitude_tx")

    # 如果腾讯振幅不存在，则使用原有高低价计算
    if amplitude is None:
        pre_close = value("pre_close", 0)
        high = value("high", 0)
        low = value("low", 0)

        if pre_close > 0:
            amplitude = (
                (high - low) / pre_close * 100
            )
        else:
            amplitude = 0

    # =========================
    # 1. 今日强度：12分
    # 理想涨幅约 +1.5%
    # =========================

    today_score = target_score(
        change,
        target=1.5,
        width=4.5,
        max_score=12
    )

    # =========================
    # 2. 5日趋势：14分
    # 理想：刚走强，但没连续暴涨
    # =========================

    d5_score = target_score(
        d5,
        target=4.0,
        width=13.0,
        max_score=14
    )

    # =========================
    # 3. 10日趋势：10分
    # =========================

    d10_score = target_score(
        d10,
        target=7.0,
        width=22.0,
        max_score=10
    )

    # =========================
    # 4. 20日位置：10分
    # =========================

    d20_score = target_score(
        d20,
        target=10.0,
        width=35.0,
        max_score=10
    )

    # =========================
    # 5. 60日位置：8分
    # 避免长期已经涨太多
    # =========================

    d60_score = target_score(
        d60,
        target=15.0,
        width=65.0,
        max_score=8
    )

    # =========================
    # 6. 量比：12分
    # 理想约 1.8
    # =========================

    volume_score = target_score(
        volume_ratio,
        target=1.8,
        width=2.5,
        max_score=12
    )

    # =========================
    # 7. 换手率：10分
    # 理想约 5%
    # =========================

    turnover_score = target_score(
        turnover_rate,
        target=5.0,
        width=10.0,
        max_score=10
    )

    # =========================
    # 8. 流动性：8分
    # 用成交额做连续评分
    # =========================

    if amount is None or amount <= 0:
        liquidity_score = 0

    else:
        liquidity_score = (
            math.log10(max(amount, 1)) - 7
        ) / 2.5 * 8

        liquidity_score = clamp(
            liquidity_score,
            0,
            8
        )

    # =========================
    # 9. 估值：6分
    # =========================

    if pe is None:
        pe_score = 3.0

    elif pe <= 0:
        pe_score = 1.0

    elif 8 <= pe <= 45:
        pe_score = (
            6
            - abs(pe - 25) / 20 * 1.5
        )

    elif 45 < pe <= 80:
        pe_score = 3.0

    elif 0 < pe < 8:
        pe_score = 4.0

    else:
        pe_score = 1.5

    pe_score = clamp(
        pe_score,
        0,
        6
    )

    # =========================
    # 10. 波动 + 涨速风险：10分
    # =========================

    amplitude_score = target_score(
        amplitude,
        target=3.5,
        width=6.5,
        max_score=6
    )

    speed_score = target_score(
        speed,
        target=0.3,
        width=2.8,
        max_score=4
    )

    risk_score = (
        amplitude_score
        + speed_score
    )

    # =========================
    # 基础总分
    # =========================

    score = (
        today_score
        + d5_score
        + d10_score
        + d20_score
        + d60_score
        + volume_score
        + turnover_score
        + liquidity_score
        + pe_score
        + risk_score
    )

    # =========================
    # 防追高扣分
    # =========================

    penalty = 0

    # 当天已经明显拉高
    if change > 5:
        penalty += 8

    if change > 7:
        penalty += 6

    # 5日涨太多
    if d5 is not None and d5 > 12:
        penalty += 6

    if d5 is not None and d5 > 18:
        penalty += 6

    # 10日过热
    if d10 is not None and d10 > 20:
        penalty += 6

    # 20日已经大幅上涨
    if d20 is not None and d20 > 35:
        penalty += 8

    # 60日高位
    if d60 is not None and d60 > 70:
        penalty += 6

    # 量比过度异常
    if (
        volume_ratio is not None
        and volume_ratio > 4
    ):
        penalty += 4

    # 换手过高
    if (
        turnover_rate is not None
        and turnover_rate > 18
    ):
        penalty += 4

    # 短时涨速过快
    if speed is not None and speed > 3:
        penalty += 5

    score -= penalty

    score = clamp(
        score,
        0,
        100
    )

    # 保留原函数返回格式，
    # 避免影响 build_results()
    return (
        round(score, 2),
        round(amplitude, 2),
        0.5
    )


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
