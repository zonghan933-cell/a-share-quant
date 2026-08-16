import json
import os
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


HISTORY_DIR = "history"
AUCTION_HISTORY_DIR = "auction_history"

BEIJING = ZoneInfo("Asia/Shanghai")

# V1 Shadow 的有效研究窗口：
# 重点只把 09:20:00 ~ 09:24:59 的数据标记为 research_valid=True。
# 9:20 后开盘集合竞价阶段不可撤单，噪声相对前半段更低。
RESEARCH_START = dt_time(9, 20, 0)
RESEARCH_END = dt_time(9, 24, 59)

# 允许脚本在更宽一点的时间窗口运行并保存诊断数据。
# 09:15~09:29 之外运行时直接跳过，避免把午盘/收盘数据混进竞价样本。
CAPTURE_START = dt_time(9, 15, 0)
CAPTURE_END = dt_time(9, 29, 59)


def normalize_code(code):
    code = str(code).strip().lower()

    if code.startswith(("sh", "sz", "bj")):
        code = code[2:]

    return code.zfill(6)


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def clean_json_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, payload):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    temp_path = f"{path}.tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temp_path,
        path
    )


def find_latest_signal_snapshot():
    """
    找到 history/ 中最新的一份收盘选股快照。
    竞价模块只读取它，不修改它。
    """
    if not os.path.isdir(HISTORY_DIR):
        return None, None

    candidates = []

    for filename in os.listdir(HISTORY_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(
            HISTORY_DIR,
            filename
        )

        try:
            payload = load_json(path)
            signal_date = str(
                payload.get(
                    "signal_date",
                    ""
                )
            ).strip()

            if not signal_date:
                continue

            candidates.append(
                (
                    signal_date,
                    path,
                    payload
                )
            )

        except Exception:
            continue

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    _, path, payload = candidates[0]

    return path, payload


def get_capture_phase(now):
    t = now.time()

    if dt_time(9, 15, 0) <= t < dt_time(9, 20, 0):
        return "竞价前半段_可撤单"

    if dt_time(9, 20, 0) <= t <= dt_time(9, 24, 59):
        return "竞价后半段_不可撤单"

    if dt_time(9, 25, 0) <= t <= dt_time(9, 29, 59):
        return "竞价结束后_开盘前后"

    return "非竞价时段"


def is_research_valid(now):
    t = now.time()

    return (
        RESEARCH_START
        <= t
        <= RESEARCH_END
    )


def gap_label(change_pct):
    """
    这里只做客观描述，不输出买入/卖出建议。
    阈值仅用于后续分组统计。
    """
    if change_pct is None:
        return "未知"

    if change_pct >= 5.0:
        return "大幅高开"

    if change_pct >= 2.0:
        return "明显高开"

    if change_pct >= 0.5:
        return "小幅高开"

    if change_pct > -0.5:
        return "近平开"

    if change_pct > -2.0:
        return "小幅低开"

    if change_pct > -5.0:
        return "明显低开"

    return "大幅低开"


def fetch_tencent_quotes():
    """
    与 scan.py 使用同一腾讯全市场实时行情源。
    V1 Shadow 暂不把任何字段解释为“真实竞价成交量”；
    先原样记录腾讯返回的核心字段，等待真实竞价样本验证。
    """
    print(
        "正在获取腾讯 A 股实时行情..."
    )

    df = ak.stock_zh_a_spot_tx()

    if df is None or df.empty:
        raise RuntimeError(
            "腾讯实时行情返回空数据"
        )

    if "code" not in df.columns:
        raise RuntimeError(
            "腾讯实时行情缺少 code 字段"
        )

    df = df.copy()

    df["_norm_code"] = (
        df["code"]
        .map(normalize_code)
    )

    return df


def build_candidate_record(
    candidate,
    quote_row
):
    code = normalize_code(
        candidate.get("code", "")
    )

    signal_close = safe_float(
        candidate.get("price")
    )

    quote_price = safe_float(
        quote_row.get("zxj")
    )

    quote_change_pct = safe_float(
        quote_row.get("zdf")
    )

    # 如果数据源没有直接涨跌幅，再用收盘快照价格做备用推算。
    derived_change_pct = None

    if (
        quote_price is not None
        and signal_close not in (
            None,
            0
        )
    ):
        derived_change_pct = (
            quote_price
            / signal_close
            - 1.0
        ) * 100.0

    effective_change_pct = (
        quote_change_pct
        if quote_change_pct is not None
        else derived_change_pct
    )

    quote_moved = None

    if (
        quote_price is not None
        and signal_close is not None
    ):
        quote_moved = (
            abs(
                quote_price
                - signal_close
            )
            > 0.000001
        )

    raw_fields = {}

    # 这些是当前 scan.py 已实际使用过的腾讯字段。
    # 保留原始值，方便第一批真实竞价样本回来后检查其含义。
    for field in (
        "code",
        "name",
        "zxj",
        "zdf",
        "volume",
        "turnover",
        "lb",
        "speed",
        "zf",
        "hsl",
        "hs1",
        "pe_ttm",
        "pn",
        "zdf_d5",
        "zdf_d10",
        "zdf_d20",
        "zdf_d60",
    ):
        if field in quote_row.index:
            raw_fields[field] = (
                clean_json_value(
                    quote_row.get(field)
                )
            )

    return {
        "signal_rank": candidate.get(
            "rank"
        ),
        "code": code,
        "name": candidate.get(
            "name"
        ),
        "signal_close": signal_close,
        "signal_score": candidate.get(
            "score"
        ),
        "signal_industry_adjustment": (
            candidate.get(
                "industry_adjustment"
            )
        ),
        "signal_combined_score": (
            candidate.get(
                "combined_score",
                candidate.get("score")
            )
        ),
        "signal_industry": candidate.get(
            "industry"
        ),
        "signal_sector_confirmation": (
            candidate.get(
                "sector_confirmation"
            )
        ),

        # Shadow 实时字段
        "quote_price": (
            round(
                quote_price,
                4
            )
            if quote_price is not None
            else None
        ),
        "quote_change_pct": (
            round(
                quote_change_pct,
                4
            )
            if quote_change_pct is not None
            else None
        ),
        "derived_change_from_signal_close_pct": (
            round(
                derived_change_pct,
                4
            )
            if derived_change_pct is not None
            else None
        ),
        "effective_gap_pct": (
            round(
                effective_change_pct,
                4
            )
            if effective_change_pct is not None
            else None
        ),
        "gap_label": gap_label(
            effective_change_pct
        ),
        "quote_moved_from_signal_close": (
            quote_moved
        ),

        # 先记录，不做交易含义解释
        "raw_volume": safe_float(
            quote_row.get("volume")
        ),
        "raw_turnover": safe_float(
            quote_row.get("turnover")
        ),
        "raw_volume_ratio": safe_float(
            quote_row.get("lb")
        ),
        "raw_speed": safe_float(
            quote_row.get("speed")
        ),

        "raw_quote_fields": raw_fields,
    }


def main():
    now = datetime.now(BEIJING)

    print("")
    print(
        "A股竞价模块 V1 Shadow"
    )
    print(
        "北京时间：",
        now.isoformat()
    )

    if now.weekday() >= 5:
        print(
            "跳过：今天是周末，"
            "不保存竞价样本。"
        )
        return

    if not (
        CAPTURE_START
        <= now.time()
        <= CAPTURE_END
    ):
        print(
            "跳过：当前不在 "
            "09:15~09:29 竞价采集窗口。"
        )
        print(
            "建议在真实交易日 "
            "09:22~09:24 手动触发。"
        )
        return

    snapshot_path, snapshot = (
        find_latest_signal_snapshot()
    )

    if snapshot is None:
        print(
            "跳过：没有找到 history/"
            "中的收盘选股快照。"
        )
        return

    stocks = snapshot.get(
        "stocks",
        []
    )

    if not stocks:
        print(
            "跳过：最新收盘快照中"
            "没有候选股票。"
        )
        return

    try:
        quotes = fetch_tencent_quotes()
    except Exception as e:
        print(
            "竞价行情获取失败：",
            repr(e)
        )
        return

    quote_map = {}

    for _, row in quotes.iterrows():
        code = row.get(
            "_norm_code"
        )

        if code:
            quote_map[code] = row

    records = []
    missing_codes = []

    for candidate in stocks:
        code = normalize_code(
            candidate.get(
                "code",
                ""
            )
        )

        row = quote_map.get(code)

        if row is None:
            missing_codes.append(code)
            continue

        records.append(
            build_candidate_record(
                candidate,
                row
            )
        )

    phase = get_capture_phase(now)
    research_valid = (
        is_research_valid(now)
    )

    payload = {
        "schema_version": "auction-shadow-1.0",
        "mode": "V1 Shadow",
        "capture_date": (
            now.date().isoformat()
        ),
        "capture_time": (
            now.isoformat()
        ),
        "capture_phase": phase,
        "research_valid": (
            research_valid
        ),
        "research_window": (
            "09:20:00-09:24:59 "
            "Asia/Shanghai"
        ),
        "signal_snapshot_file": (
            snapshot_path
        ),
        "signal_date": snapshot.get(
            "signal_date"
        ),
        "signal_strategy": snapshot.get(
            "strategy"
        ),
        "candidate_count": len(
            stocks
        ),
        "matched_count": len(
            records
        ),
        "missing_count": len(
            missing_codes
        ),
        "missing_codes": missing_codes,
        "data_source": (
            "腾讯A股实时行情 "
            "stock_zh_a_spot_tx"
        ),
        "important_note": (
            "Shadow阶段：当前仅采集并冻结竞价时点数据；"
            "不产生买入/卖出信号。"
            "raw_volume/raw_turnover 等字段的竞价含义"
            "需用真实样本验证后再进入评分。"
        ),
        "stocks": records,
    }

    date_dir = os.path.join(
        AUCTION_HISTORY_DIR,
        now.date().isoformat()
    )

    filename = (
        now.strftime("%H%M%S")
        + ".json"
    )

    output_path = os.path.join(
        date_dir,
        filename
    )

    try:
        save_json_atomic(
            output_path,
            payload
        )
    except Exception as e:
        print(
            "竞价快照保存失败：",
            repr(e)
        )
        return

    print("")
    print(
        "竞价快照已保存：",
        output_path
    )
    print(
        "阶段：",
        phase
    )
    print(
        "是否进入正式研究样本：",
        research_valid
    )
    print(
        "来源收盘信号日：",
        snapshot.get("signal_date")
    )
    print(
        "候选数：",
        len(stocks)
    )
    print(
        "匹配数：",
        len(records)
    )
    print(
        "缺失数：",
        len(missing_codes)
    )

    print("")
    print(
        "竞价 Shadow Top 10："
    )

    for stock in records[:10]:
        gap = stock.get(
            "effective_gap_pct"
        )

        gap_text = (
            f"{gap:+.2f}%"
            if gap is not None
            else "未知"
        )

        print(
            f"{int(stock.get('signal_rank') or 0):02d}. "
            f"{stock.get('code')} "
            f"{stock.get('name')} | "
            f"昨收快照={stock.get('signal_close')} | "
            f"实时价={stock.get('quote_price')} | "
            f"竞价变化={gap_text} | "
            f"标签={stock.get('gap_label')} | "
            f"综合分={stock.get('signal_combined_score')}"
        )


if __name__ == "__main__":
    main()
