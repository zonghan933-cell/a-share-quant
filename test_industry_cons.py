import time
import re

import akshare as ak
import pandas as pd


TEST_STOCKS = [
    ("002349", "精华制药"),
    ("002033", "丽江股份"),
]


def to_number(value):
    """
    把 3.71% / -1.2 / None 等统一转成数字
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    text = text.replace("%", "")

    try:
        return float(text)
    except Exception:
        return None


def normalize_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    text = (
        text
        .replace("、", "")
        .replace("，", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("（", "(")
        .replace("）", ")")
    )

    return text


def get_fund_table():
    """
    取得同花顺即时 + 3日行业资金流
    """

    print("正在获取同花顺行业资金流...")

    start = time.time()

    now_df = ak.stock_fund_flow_industry(
        symbol="即时"
    )

    three_df = ak.stock_fund_flow_industry(
        symbol="3日排行"
    )

    print(
        f"行业资金流获取完成："
        f"{time.time() - start:.2f} 秒"
    )

    # -------------------------
    # 即时
    # -------------------------

    now = now_df.copy()

    now["industry"] = (
        now["行业"]
        .astype(str)
        .str.strip()
    )

    now["change_now"] = (
        now["行业-涨跌幅"]
        .apply(to_number)
    )

    now["net_now"] = (
        now["净额"]
        .apply(to_number)
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

    # -------------------------
    # 3日
    # -------------------------

    three = three_df.copy()

    three["industry"] = (
        three["行业"]
        .astype(str)
        .str.strip()
    )

    three["change_3d"] = (
        three["阶段涨跌幅"]
        .apply(to_number)
    )

    three["net_3d"] = (
        three["净额"]
        .apply(to_number)
    )

    three = three[
        [
            "industry",
            "change_3d",
            "net_3d",
        ]
    ]

    # -------------------------
    # 合并
    # -------------------------

    table = pd.merge(
        now,
        three,
        on="industry",
        how="left"
    )

    # -------------------------
    # 百分位排名
    # -------------------------

    table["rank_change_now"] = (
        table["change_now"]
        .rank(pct=True)
    )

    table["rank_change_3d"] = (
        table["change_3d"]
        .rank(pct=True)
    )

    table["rank_net_now"] = (
        table["net_now"]
        .rank(pct=True)
    )

    table["rank_net_3d"] = (
        table["net_3d"]
        .rank(pct=True)
    )

    # -------------------------
    # 板块热度 0~10
    # -------------------------

    table["heat_score"] = (
        table["rank_change_now"] * 0.35
        + table["rank_change_3d"] * 0.30
        + table["rank_net_now"] * 0.20
        + table["rank_net_3d"] * 0.15
    ) * 10

    return table


def classification_priority(text):
    """
    分类标准优先级
    """

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


def keyword_match(candidate, industries):
    """
    对名称不完全一致的行业做兼容匹配
    """

    candidate = normalize_text(candidate)

    # 中药
    if (
        "中药" in candidate
        or "中成药" in candidate
    ):
        for industry in industries:
            if "中药" in industry:
                return industry

    # 旅游 / 景区
    if any(
        word in candidate
        for word in [
            "景点",
            "景区",
            "旅游",
            "自然景区",
        ]
    ):
        for industry in industries:
            if (
                "旅游" in industry
                or "景区" in industry
            ):
                return industry

    # 酒店
    if any(
        word in candidate
        for word in [
            "酒店",
            "餐馆",
            "餐饮",
        ]
    ):
        for industry in industries:
            if (
                "酒店" in industry
                or "餐饮" in industry
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

    return None


def match_stock_industry(code, industry_table):
    """
    自动寻找个股最适合的同花顺行业
    """

    df = ak.stock_industry_change_cninfo(
        symbol=code,
        start_date="20000101",
        end_date="20260814"
    )

    if df is None or df.empty:
        return None, None, None

    df = df.copy()

    df["priority"] = (
        df["分类标准"]
        .apply(classification_priority)
    )

    df["变更日期"] = pd.to_datetime(
        df["变更日期"],
        errors="coerce"
    )

    # 优先分类标准 + 新记录
    df = df.sort_values(
        ["priority", "变更日期"],
        ascending=[False, False]
    )

    industries = (
        industry_table["industry"]
        .dropna()
        .astype(str)
        .tolist()
    )

    normalized_industries = {
        normalize_text(x): x
        for x in industries
    }

    # 行业中类优先
    fields = [
        "行业中类",
        "行业大类",
        "行业次类",
        "行业门类",
    ]

    # =========================
    # 第一轮：完全匹配
    # =========================

    for _, row in df.iterrows():

        for field in fields:

            value = row.get(field)

            if pd.isna(value):
                continue

            key = normalize_text(value)

            if key in normalized_industries:

                return (
                    normalized_industries[key],
                    field,
                    value
                )

    # =========================
    # 第二轮：包含匹配
    # =========================

    for _, row in df.iterrows():

        for field in fields:

            value = row.get(field)

            if pd.isna(value):
                continue

            candidate = normalize_text(value)

            if len(candidate) < 2:
                continue

            for industry in industries:

                target = normalize_text(industry)

                if (
                    candidate == target
                    or candidate in target
                    or target in candidate
                ):

                    return (
                        industry,
                        field,
                        value
                    )

    # =========================
    # 第三轮：关键词兼容
    # =========================

    for _, row in df.iterrows():

        for field in fields:

            value = row.get(field)

            if pd.isna(value):
                continue

            matched = keyword_match(
                value,
                industries
            )

            if matched:

                return (
                    matched,
                    field,
                    value
                )

    return None, None, None


def main():

    print("=" * 70)
    print("V2.3 行业自动匹配 + 板块热度测试")
    print("=" * 70)

    industry_table = get_fund_table()

    print("")
    print("=" * 70)
    print("当前行业热度 Top 10")
    print("=" * 70)

    top = (
        industry_table
        .sort_values(
            "heat_score",
            ascending=False
        )
        .head(10)
    )

    for _, row in top.iterrows():

        print(
            f"{row['industry']} | "
            f"热度={row['heat_score']:.2f} | "
            f"今日={row['change_now']}% | "
            f"3日={row['change_3d']}% | "
            f"今日净额={row['net_now']}亿 | "
            f"3日净额={row['net_3d']}亿"
        )

    print("")
    print("=" * 70)
    print("个股行业匹配")
    print("=" * 70)

    for code, name in TEST_STOCKS:

        start = time.time()

        try:

            industry, field, raw_value = (
                match_stock_industry(
                    code,
                    industry_table
                )
            )

            elapsed = time.time() - start

            print("")
            print(
                f"{code} {name}"
            )

            print(
                f"耗时: {elapsed:.2f} 秒"
            )

            print(
                f"匹配行业: {industry}"
            )

            print(
                f"匹配来源字段: {field}"
            )

            print(
                f"原始分类: {raw_value}"
            )

            if industry is not None:

                row = industry_table[
                    industry_table["industry"]
                    == industry
                ]

                if not row.empty:

                    row = row.iloc[0]

                    print(
                        f"板块热度: "
                        f"{row['heat_score']:.2f}/10"
                    )

                    print(
                        f"今日涨幅: "
                        f"{row['change_now']}%"
                    )

                    print(
                        f"3日涨幅: "
                        f"{row['change_3d']}%"
                    )

                    print(
                        f"今日净额: "
                        f"{row['net_now']}亿"
                    )

                    print(
                        f"3日净额: "
                        f"{row['net_3d']}亿"
                    )

        except Exception as e:

            print("")
            print(
                f"{code} {name} ERROR"
            )

            print(repr(e))


if __name__ == "__main__":
    main()
