import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


LATEST_FILE = "latest.json"
HISTORY_DIR = "history"

BEIJING = ZoneInfo("Asia/Shanghai")

MARKET_DATE_LOOKBACK_DAYS = 20
MARKET_DATE_QUERY_TIMEOUT = 8


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


def normalize_code(code):
    code = str(code).strip().lower()

    if code.startswith(
        ("sh", "sz", "bj")
    ):
        code = code[2:]

    return code.zfill(6)


def to_tx_symbol(code):
    code = normalize_code(code)

    if code.startswith("6"):
        return f"sh{code}"

    return f"sz{code}"


def resolve_reference_date(
    latest_payload
):
    """
    优先使用 latest.json 自己的数据时间，
    避免脚本实际运行时间和行情快照日期不一致。
    """
    raw_time = latest_payload.get(
        "data_time"
    )

    if raw_time:
        try:
            dt = datetime.fromisoformat(
                raw_time
            )

            if dt.tzinfo is None:
                dt = dt.replace(
                    tzinfo=BEIJING
                )
            else:
                dt = dt.astimezone(
                    BEIJING
                )

            return dt.date()

        except Exception:
            pass

    return datetime.now(
        BEIJING
    ).date()


def resolve_market_date_by_calendar(
    reference_date
):
    """
    第一方案：
    使用股票交易日历确认真实交易日。

    这样不会依赖当日日K是否已经更新，
    也能正确处理周末和节假日。
    """
    df = ak.tool_trade_date_hist_sina()

    if df is None or df.empty:
        raise RuntimeError(
            "交易日历返回空数据"
        )

    if "trade_date" not in df.columns:
        raise RuntimeError(
            "交易日历缺少 trade_date 字段"
        )

    dates = pd.to_datetime(
        df["trade_date"],
        errors="coerce"
    ).dropna()

    if dates.empty:
        raise RuntimeError(
            "交易日历没有有效日期"
        )

    valid_dates = [
        item.date()
        for item in dates
        if item.date() <= reference_date
    ]

    if not valid_dates:
        raise RuntimeError(
            "交易日历没有找到有效交易日"
        )

    market_date = max(
        valid_dates
    )

    return market_date.isoformat()


def resolve_market_date_by_tx(
    stocks,
    reference_date
):
    """
    第二方案：
    如果交易日历接口失败，
    再退回原来的腾讯历史日线确认。
    """
    start_date = (
        reference_date
        - timedelta(
            days=MARKET_DATE_LOOKBACK_DAYS
        )
    ).strftime("%Y%m%d")

    end_date = reference_date.strftime(
        "%Y%m%d"
    )

    errors = []

    for stock in stocks[:3]:
        code = normalize_code(
            stock.get(
                "code",
                ""
            )
        )

        if not code:
            continue

        symbol = to_tx_symbol(
            code
        )

        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="",
                timeout=MARKET_DATE_QUERY_TIMEOUT
            )

            if df is None or df.empty:
                errors.append(
                    f"{code}: 空数据"
                )
                continue

            if "date" not in df.columns:
                errors.append(
                    f"{code}: 缺少 date 字段"
                )
                continue

            dates = pd.to_datetime(
                df["date"],
                errors="coerce"
            ).dropna()

            if dates.empty:
                errors.append(
                    f"{code}: 无有效日期"
                )
                continue

            market_date = (
                dates.max().date()
            )

            return market_date.isoformat()

        except Exception as e:
            errors.append(
                f"{code}: "
                f"{type(e).__name__}: {e}"
            )

    raise RuntimeError(
        "腾讯历史行情也无法确认交易日；"
        + " | ".join(errors)
    )


def resolve_market_date(
    latest_payload,
    stocks
):
    """
    最终交易日确认逻辑：

    1. 使用 latest.json 的 data_time
    2. 优先交易日历
    3. 交易日历失败才使用腾讯历史日线
    """
    reference_date = (
        resolve_reference_date(
            latest_payload
        )
    )

    print(
        "绩效记录参考日期：",
        reference_date.isoformat()
    )

    try:
        market_date = (
            resolve_market_date_by_calendar(
                reference_date
            )
        )

        print(
            "交易日历确认成功：",
            market_date
        )

        return market_date

    except Exception as e:
        print(
            "交易日历确认失败，"
            "切换腾讯历史行情：",
            repr(e)
        )

    return resolve_market_date_by_tx(
        stocks,
        reference_date
    )


def build_snapshot(
    latest_payload,
    signal_date
):
    stocks = latest_payload.get(
        "stocks",
        []
    )

    snapshot_stocks = []

    for rank, stock in enumerate(
        stocks,
        start=1
    ):
        item = dict(stock)
        item["rank"] = rank
        snapshot_stocks.append(
            item
        )

    return {
        "schema_version": "1.0",
        "signal_date": signal_date,
        "recorded_at": datetime.now(
            BEIJING
        ).isoformat(),
        "strategy": latest_payload.get(
            "strategy"
        ),
        "data_source": latest_payload.get(
            "data_source"
        ),
        "data_quality": latest_payload.get(
            "data_quality"
        ),
        "source_data_time": (
            latest_payload.get(
                "data_time"
            )
        ),
        "source_calculated_at": (
            latest_payload.get(
                "calculated_at"
            )
        ),
        "stock_count": len(
            snapshot_stocks
        ),
        "stocks": snapshot_stocks,
    }


def print_snapshot_summary(
    snapshot,
    path
):
    print("")
    print("绩效快照已保存")

    print(
        "交易日：",
        snapshot["signal_date"]
    )

    print(
        "策略：",
        snapshot.get(
            "strategy"
        )
    )

    print(
        "股票数量：",
        snapshot["stock_count"]
    )

    print(
        "文件：",
        path
    )

    print("")
    print("快照 Top 10：")

    for stock in snapshot[
        "stocks"
    ][:10]:

        print(
            f"{stock['rank']:02d}. "
            f"{stock.get('code', '')} "
            f"{stock.get('name', '')} | "
            f"原始分="
            f"{stock.get('score')} | "
            f"行业调整="
            f"{stock.get(
                'industry_adjustment',
                0.0
            ):+.2f} | "
            f"综合分="
            f"{stock.get(
                'combined_score',
                stock.get('score')
            )}"
        )


def main():
    if not os.path.exists(
        LATEST_FILE
    ):
        print(
            "绩效记录跳过："
            f"未找到 {LATEST_FILE}"
        )
        return

    try:
        latest_payload = load_json(
            LATEST_FILE
        )

    except Exception as e:
        print(
            "绩效记录跳过："
            "latest.json 读取失败：",
            repr(e)
        )
        return

    if latest_payload.get(
        "status"
    ) != "ok":

        print(
            "绩效记录跳过："
            "本次扫描状态不是 ok"
        )
        return

    stocks = latest_payload.get(
        "stocks",
        []
    )

    if not stocks:
        print(
            "绩效记录跳过："
            "本次扫描没有股票结果"
        )
        return

    try:
        signal_date = (
            resolve_market_date(
                latest_payload,
                stocks
            )
        )

    except Exception as e:
        print(
            "绩效记录跳过："
            "无法确认真实交易日：",
            repr(e)
        )
        return

    os.makedirs(
        HISTORY_DIR,
        exist_ok=True
    )

    snapshot_path = os.path.join(
        HISTORY_DIR,
        f"{signal_date}.json"
    )

    # 同一个交易日只保存第一次快照。
    # 后续重复运行绝不覆盖，
    # 防止回看偏差。
    if os.path.exists(
        snapshot_path
    ):
        print("")
        print(
            "绩效快照已存在，"
            "保持首次结果，不覆盖："
        )
        print(
            snapshot_path
        )
        return

    snapshot = build_snapshot(
        latest_payload,
        signal_date
    )

    try:
        save_json_atomic(
            snapshot_path,
            snapshot
        )

    except Exception as e:
        print(
            "绩效快照保存失败：",
            repr(e)
        )
        return

    print_snapshot_summary(
        snapshot,
        snapshot_path
    )


if __name__ == "__main__":
    main()
