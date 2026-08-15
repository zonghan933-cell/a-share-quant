import json
import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd


OUTPUT_FILE = "latest.json"
INDUSTRY_CACHE_FILE = "industry_cache.json"
MIN_PRICE = 3
MAX_PRICE = 100

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
    (df["price"] >= MIN_PRICE)
    & (df["price"] <= MAX_PRICE)
    & (df["pre_close"] > 0)
    & (df["amount"].fillna(0) > 0)
]

    return df


def score_stock(row):
    """
    V2.2 量化评分
    核心：
    - 偏好温和启动
    - 更严格限制20/60日高位
    - 量价、换手、估值、波动共同评分
    - 返回详细评分拆解

    满分100
    """

    def value(name, default=None):
        return safe_float(row.get(name), default)

    def target_score(x, target, width, max_score, neutral=0.45):
        """
        越接近理想值越高分。
        数据缺失给中性分，不直接记0。
        """
        if x is None:
            return max_score * neutral

        distance = abs(x - target)

        score = max_score * (
            1 - distance / width
        )

        return clamp(score, 0, max_score)

    # =========================
    # 数据
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

    if amplitude is None:
        pre_close = value("pre_close", 0)
        high = value("high", 0)
        low = value("low", 0)

        if pre_close > 0:
            amplitude = (
                (high - low)
                / pre_close
                * 100
            )
        else:
            amplitude = 0

    # =========================
    # 1. 今日强度 12分
    # 理想约 +1.5%
    # =========================

    today_score = target_score(
        change,
        target=1.5,
        width=4.5,
        max_score=12
    )

    # =========================
    # 2. 5日趋势 14分
    # =========================

    d5_score = target_score(
        d5,
        target=4.0,
        width=12.0,
        max_score=14
    )

    # =========================
    # 3. 10日趋势 10分
    # =========================

    d10_score = target_score(
        d10,
        target=7.0,
        width=20.0,
        max_score=10
    )

    # =========================
    # 4. 20日位置 10分
    # V2.2更偏低位
    # =========================

    d20_score = target_score(
        d20,
        target=8.0,
        width=25.0,
        max_score=10
    )

    # =========================
    # 5. 60日位置 8分
    # =========================

    d60_score = target_score(
        d60,
        target=12.0,
        width=45.0,
        max_score=8
    )

    # =========================
    # 6. 量比 12分
    # 理想1.5~2附近
    # =========================

    volume_score = target_score(
        volume_ratio,
        target=1.8,
        width=2.5,
        max_score=12
    )

    # =========================
    # 7. 换手率 10分
    # =========================

    turnover_score = target_score(
        turnover_rate,
        target=5.0,
        width=9.0,
        max_score=10
    )

    # =========================
    # 8. 流动性 8分
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
    # 9. 估值 6分
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
    # 10. 风险控制 10分
    # =========================

    amplitude_score = target_score(
        amplitude,
        target=3.5,
        width=6.0,
        max_score=6
    )

    speed_score = target_score(
        speed,
        target=0.3,
        width=2.5,
        max_score=4
    )

    risk_score = (
        amplitude_score
        + speed_score
    )

    # =========================
    # 基础分
    # =========================

    base_score = (
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

    penalty_today = 0
    penalty_d5 = 0
    penalty_d10 = 0
    penalty_d20 = 0
    penalty_d60 = 0
    penalty_volume = 0
    penalty_turnover = 0
    penalty_speed = 0

    # 当天过热
    if change > 5:
        penalty_today += 5

    if change > 7:
        penalty_today += 5

    # 5日涨幅过热
    if d5 is not None and d5 > 10:
        penalty_d5 += 3

    if d5 is not None and d5 > 15:
        penalty_d5 += 5

    if d5 is not None and d5 > 22:
        penalty_d5 += 6

    # 10日过热
    if d10 is not None and d10 > 18:
        penalty_d10 += 4

    if d10 is not None and d10 > 28:
        penalty_d10 += 6

    # =========================
    # V2.2核心：
    # 20日高位更严格
    # =========================

    if d20 is not None and d20 > 18:
        penalty_d20 += 3

    if d20 is not None and d20 > 25:
        penalty_d20 += 4

    if d20 is not None and d20 > 35:
        penalty_d20 += 6

    # =========================
    # 60日高位
    # =========================

    if d60 is not None and d60 > 35:
        penalty_d60 += 3

    if d60 is not None and d60 > 50:
        penalty_d60 += 5

    if d60 is not None and d60 > 75:
        penalty_d60 += 7

    # 量比异常
    if (
        volume_ratio is not None
        and volume_ratio > 4
    ):
        penalty_volume += 4

    if (
        volume_ratio is not None
        and volume_ratio > 7
    ):
        penalty_volume += 4

    # 换手异常
    if (
        turnover_rate is not None
        and turnover_rate > 18
    ):
        penalty_turnover += 4

    if (
        turnover_rate is not None
        and turnover_rate > 28
    ):
        penalty_turnover += 5

    # 涨速异常
    if speed is not None and speed > 3:
        penalty_speed += 4

    if speed is not None and speed > 5:
        penalty_speed += 5

    total_penalty = (
        penalty_today
        + penalty_d5
        + penalty_d10
        + penalty_d20
        + penalty_d60
        + penalty_volume
        + penalty_turnover
        + penalty_speed
    )

    final_score = (
        base_score
        - total_penalty
    )

    final_score = clamp(
        final_score,
        0,
        100
    )

    # =========================
    # 评分拆解
    # =========================

    score_detail = {
        "today": round(today_score, 2),
        "d5": round(d5_score, 2),
        "d10": round(d10_score, 2),
        "d20": round(d20_score, 2),
        "d60": round(d60_score, 2),

        "volume_ratio": round(
            volume_score, 2
        ),

        "turnover": round(
            turnover_score, 2
        ),

        "liquidity": round(
            liquidity_score, 2
        ),

        "valuation": round(
            pe_score, 2
        ),

        "risk": round(
            risk_score, 2
        ),

        "base_score": round(
            base_score, 2
        ),

        "penalty_today": penalty_today,
        "penalty_d5": penalty_d5,
        "penalty_d10": penalty_d10,
        "penalty_d20": penalty_d20,
        "penalty_d60": penalty_d60,
        "penalty_volume": penalty_volume,
        "penalty_turnover": penalty_turnover,
        "penalty_speed": penalty_speed,

        "total_penalty": total_penalty,

        "final_score": round(
            final_score, 2
        )
    }

    return (
        round(final_score, 2),
        round(amplitude, 2),
        0.5,
        score_detail
    )

# =========================================================
# V2.3A 行业热度 / 行业缓存
# 只用于解释和确认，暂时不改变个股原始 score
# =========================================================


def industry_to_number(value):
    if value is None:
        return None

    text = str(value).strip()

    if (
        not text
        or text.lower() == "nan"
        or text == "--"
    ):
        return None

    text = (
        text
        .replace("%", "")
        .replace("亿", "")
        .replace(",", "")
    )

    try:
        return float(text)
    except Exception:
        return None


def normalize_industry_text(value):
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("、", "")
        .replace("，", "")
        .replace(",", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def load_industry_cache():
    try:
        with open(
            INDUSTRY_CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, dict):
                return data

    except Exception:
        pass

    return {}


def save_industry_cache(cache):
    try:
        with open(
            INDUSTRY_CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                cache,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:
        print(
            "行业缓存保存失败:",
            repr(e)
        )


def get_industry_heat_table():
    """
    同花顺：
    即时行业资金流 + 3日行业资金流

    返回90个行业的板块热度。
    """

    print("")
    print("正在获取行业资金热度...")

    start = time.time()

    now_df = ak.stock_fund_flow_industry(
        symbol="即时"
    )

    three_df = ak.stock_fund_flow_industry(
        symbol="3日排行"
    )

    now = now_df.copy()

    now["industry"] = (
        now["行业"]
        .astype(str)
        .str.strip()
    )

    now["change_now"] = (
        now["行业-涨跌幅"]
        .apply(industry_to_number)
    )

    now["net_now"] = (
        now["净额"]
        .apply(industry_to_number)
    )

    now = now[
        [
            "industry",
            "change_now",
            "net_now",
            "公司家数",
            "领涨股",
            "领涨股-涨跌幅",
        ]
    ]

    three = three_df.copy()

    three["industry"] = (
        three["行业"]
        .astype(str)
        .str.strip()
    )

    three["change_3d"] = (
        three["阶段涨跌幅"]
        .apply(industry_to_number)
    )

    three["net_3d"] = (
        three["净额"]
        .apply(industry_to_number)
    )

    three = three[
        [
            "industry",
            "change_3d",
            "net_3d",
        ]
    ]

    table = pd.merge(
        now,
        three,
        on="industry",
        how="left"
    )

    # 用全市场90个行业做百分位排名
    table["rank_change_now"] = (
        table["change_now"]
        .rank(pct=True)
        .fillna(0.5)
    )

    table["rank_change_3d"] = (
        table["change_3d"]
        .rank(pct=True)
        .fillna(0.5)
    )

    table["rank_net_now"] = (
        table["net_now"]
        .rank(pct=True)
        .fillna(0.5)
    )

    table["rank_net_3d"] = (
        table["net_3d"]
        .rank(pct=True)
        .fillna(0.5)
    )

    # 热度满分10
    table["heat_score"] = (
        table["rank_change_now"] * 0.35
        + table["rank_change_3d"] * 0.30
        + table["rank_net_now"] * 0.20
        + table["rank_net_3d"] * 0.15
    ) * 10

    elapsed = time.time() - start

    print(
        f"行业资金热度获取成功，"
        f"耗时 {elapsed:.2f} 秒"
    )

    return table


def industry_classification_priority(text):
    text = str(text)

    if "中证行业分类标准" in text:
        return 100

    if "巨潮行业分类标准" in text:
        return 90

    if "申银万国行业分类标准" in text:
        return 80

    if "新财富行业分类标准" in text:
        return 70

    if "证监会行业分类标准" in text:
        return 50

    if "上市公司协会" in text:
        return 20

    return 10


def industry_keyword_match(
    candidate,
    industries
):
    candidate = normalize_industry_text(
        candidate
    )

    # 中药
    if (
        "中药" in candidate
        or "中成药" in candidate
    ):
        for industry in industries:
            if "中药" in industry:
                return industry

    # 旅游 / 景区 / 酒店
    if any(
        x in candidate
        for x in [
            "旅游",
            "景区",
            "景点",
            "自然景区",
            "酒店",
            "餐饮",
            "餐馆",
        ]
    ):
        for industry in industries:
            if (
                "旅游" in industry
                or "酒店" in industry
                or "景区" in industry
            ):
                return industry

    # 医药商业
    if (
        "医药商业" in candidate
        or "医药流通" in candidate
    ):
        for industry in industries:
            if "医药商业" in industry:
                return industry

    # 生物制品
    if (
        "生物制品" in candidate
        or "生物药" in candidate
    ):
        for industry in industries:
            if "生物制品" in industry:
                return industry

    # 通信设备
    if "通信设备" in candidate:
        for industry in industries:
            if "通信设备" in industry:
                return industry

    # 通信服务
    if "通信服务" in candidate:
        for industry in industries:
            if "通信服务" in industry:
                return industry

    # 半导体
    if "半导体" in candidate:
        for industry in industries:
            if "半导体" in industry:
                return industry

    return None


def match_stock_industry(
    code,
    industry_table,
    cache
):
    """
    给股票匹配同花顺90行业。

    优先缓存；
    缓存没有时才访问巨潮。
    """

    code = str(code).zfill(6)

    industries = (
        industry_table["industry"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # -------------------------
    # 1. 先查缓存
    # -------------------------

    cached = cache.get(code)

    if (
        cached
        and cached in industries
    ):
        return (
            cached,
            "cache",
            cached
        )

    # -------------------------
    # 2. 巨潮查询
    # -------------------------

    today_str = datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y%m%d")

    df = ak.stock_industry_change_cninfo(
        symbol=code,
        start_date="20000101",
        end_date=today_str
    )

    if df is None or df.empty:
        return None, None, None

    df = df.copy()

    df["priority"] = (
        df["分类标准"]
        .apply(
            industry_classification_priority
        )
    )

    df["变更日期"] = pd.to_datetime(
        df["变更日期"],
        errors="coerce"
    )

    df = df.sort_values(
        [
            "priority",
            "变更日期",
        ],
        ascending=[
            False,
            False,
        ]
    )

    normalized_industries = {
        normalize_industry_text(x): x
        for x in industries
    }

    fields = [
        "行业中类",
        "行业大类",
        "行业次类",
        "行业门类",
    ]

    # -------------------------
    # 3. 完全匹配
    # -------------------------

    for _, row in df.iterrows():

        for field in fields:

            value = row.get(field)

            if pd.isna(value):
                continue

            key = normalize_industry_text(
                value
            )

            if key in normalized_industries:

                industry = (
                    normalized_industries[key]
                )

                cache[code] = industry

                return (
                    industry,
                    field,
                    value
                )

    # -------------------------
    # 4. 包含匹配
    # -------------------------

    for _, row in df.iterrows():

        for field in fields:

            value = row.get(field)

            if pd.isna(value):
                continue

            candidate = (
                normalize_industry_text(
                    value
                )
            )

            if len(candidate) < 2:
                continue

            for industry in industries:

                target = (
                    normalize_industry_text(
                        industry
                    )
                )

                if (
                    candidate == target
                    or candidate in target
                    or target in candidate
                ):

                    cache[code] = industry

                    return (
                        industry,
                        field,
                        value
                    )

    # -------------------------
    # 5. 关键词兼容
    # -------------------------

    for _, row in df.iterrows():

        for field in fields:

            value = row.get(field)

            if pd.isna(value):
                continue

            matched = (
                industry_keyword_match(
                    value,
                    industries
                )
            )

            if matched:

                cache[code] = matched

                return (
                    matched,
                    field,
                    value
                )

    return None, None, None


def industry_confirmation(heat):
    if heat is None:
        return "未知"

    if heat >= 8:
        return "强"

    if heat >= 6:
        return "偏强"

    if heat >= 4:
        return "中性"

    return "弱"

def calculate_industry_adjustment(heat):
    """
    V2.3B Shadow

    行业只作为确认因子，不允许主导个股评分。

    中性热度 = 5
    强行业最多 +2.0
    弱行业最多 -2.5
    """

    if heat is None:
        return 0.0

    adjustment = (
        (heat - 5.0) * 0.35
    )

    adjustment = max(
        -1.8,
        min(1.2, adjustment)
    )

    return round(
        adjustment,
        2
    )
def enrich_industry_info(records):
    """
    只处理最终Top20。
    V2.3A不修改 score、不重新排序。
    """

    if not records:
        return records

    print("")
    print("=" * 60)
    print("开始进行行业板块确认")
    print("=" * 60)

    try:
        industry_table = (
            get_industry_heat_table()
        )

    except Exception as e:

        print(
            "行业资金数据获取失败:",
            repr(e)
        )

        # 行业数据失败也不能影响正常扫描
        return records

    cache = load_industry_cache()

    for item in records:

        code = str(
            item.get("code", "")
        ).zfill(6)

        name = item.get(
            "name",
            ""
        )

        try:

            industry, source_field, raw_class = (
                match_stock_industry(
                    code,
                    industry_table,
                    cache
                )
            )

        except Exception as e:

            print(
                f"{code} {name} "
                f"行业查询失败:",
                repr(e)
            )

            industry = None
            source_field = None
            raw_class = None

        item["industry"] = industry
        item["industry_match_field"] = (
            source_field
        )
        item["industry_raw_class"] = (
            raw_class
        )

        if industry is None:

            item["industry_heat"] = None
            item["industry_change_now"] = None
            item["industry_change_3d"] = None
            item["industry_net_now"] = None
            item["industry_net_3d"] = None
            item["sector_confirmation"] = (
                "未知"
            )

            print(
                f"{code} {name} "
                f"=> 未匹配行业"
            )

            continue

        matched_row = industry_table[
            industry_table["industry"]
            == industry
        ]

        if matched_row.empty:
            continue

        row = matched_row.iloc[0]

        heat = safe_float(
            row["heat_score"]
        )

        item["industry_heat"] = (
            round(heat, 2)
            if heat is not None
            else None
        )

        item["industry_change_now"] = (
            safe_float(
                row["change_now"]
            )
        )

        item["industry_change_3d"] = (
            safe_float(
                row["change_3d"]
            )
        )

        item["industry_net_now"] = (
            safe_float(
                row["net_now"]
            )
        )

        item["industry_net_3d"] = (
            safe_float(
                row["net_3d"]
            )
        )

        item["sector_confirmation"] = (
            industry_confirmation(
                heat
            )
        )
        industry_adjustment = calculate_industry_adjustment(heat)

        item["industry_adjustment"] = industry_adjustment

        base_score = item.get("score", 0.0)

        item["combined_score"] = round(
            float(base_score) + industry_adjustment,
            2
        )
        print(
            f"{code} {name} "
            f"=> {industry} | "
            f"热度={item['industry_heat']} | "
            f"确认={item['sector_confirmation']} | "
            f"原始分={base_score} | "
            f"行业调整={item['industry_adjustment']:+.2f} | "
            f"模拟综合分={item['combined_score']}"
        )

    save_industry_cache(cache)

    return records
def build_results(df):
    records = []

    for _, row in df.iterrows():
        score, amplitude, intraday_position, score_detail = score_stock(row)

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
            "speed": safe_float(row.get("speed")),
            "zdf_d5": safe_float(row.get("zdf_d5")),
            "zdf_d10": safe_float(row.get("zdf_d10")),
            "zdf_d20": safe_float(row.get("zdf_d20")),
            "zdf_d60": safe_float(row.get("zdf_d60")),
            "amplitude_tx": safe_float(row.get("amplitude_tx")),
            
            "amplitude": amplitude,
            "intraday_position": intraday_position,
            "score": score,
            "score_detail": score_detail,
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

    
        # V2.3B：
    # 先用个股原始评分筛出 Top50 候选，
    # 再加入行业确认因子，最后按综合分选出 Top20。
    top_candidates = records[:50]

    top_candidates = enrich_industry_info(
        top_candidates
    )

    top_candidates.sort(
        key=lambda x: (
            x.get(
                "combined_score",
                x.get("score", 0.0)
            ),
            x["amount"] if x["amount"] is not None else 0
        ),
        reverse=True
    )

    top_records = top_candidates[:20]

    return top_records

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
            "strategy": "主板非ST·偏低位·行业确认 V2.3B",
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
                f"涨幅={stock['change_pct']}% | "
                f"原始分={stock['score']} | "
                f"行业调整={stock.get('industry_adjustment', 0.0):+.2f} | "
                f"综合分={stock.get('combined_score', stock['score'])} | "
                f"行业={stock.get('industry', '未知')} | "
                f"确认={stock.get('sector_confirmation', '未知')}"
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
