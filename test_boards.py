import inspect
import multiprocessing as mp
import time
import traceback

import akshare as ak


TIMEOUT_SECONDS = 45


def worker(func_name, queue):
    start = time.time()

    try:
        if not hasattr(ak, func_name):
            queue.put({
                "source": func_name,
                "status": "missing",
                "seconds": 0
            })
            return

        func = getattr(ak, func_name)

        sig = inspect.signature(func)

        required_params = [
            p.name
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]

        if required_params:
            queue.put({
                "source": func_name,
                "status": "needs_args",
                "required_params": required_params,
                "signature": str(sig),
                "seconds": round(time.time() - start, 2),
            })
            return

        df = func()

        elapsed = round(time.time() - start, 2)

        if df is None:
            queue.put({
                "source": func_name,
                "status": "error",
                "reason": "返回 None",
                "seconds": elapsed,
            })
            return

        queue.put({
            "source": func_name,
            "status": "success",
            "rows": len(df),
            "columns": list(df.columns),
            "seconds": elapsed,
            "sample": df.head(5).to_dict("records"),
        })

    except Exception as e:
        queue.put({
            "source": func_name,
            "status": "error",
            "reason": repr(e),
            "traceback": traceback.format_exc()[-1500:],
            "seconds": round(time.time() - start, 2),
        })


def test_source(func_name):
    print("")
    print("=" * 70)
    print(f"开始测试: {func_name}")
    print("=" * 70)

    queue = mp.Queue()

    process = mp.Process(
        target=worker,
        args=(func_name, queue),
    )

    process.start()
    process.join(TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join()

        result = {
            "source": func_name,
            "status": "timeout",
            "seconds": TIMEOUT_SECONDS,
        }

        print(result)
        return result

    if queue.empty():
        result = {
            "source": func_name,
            "status": "error",
            "reason": "没有返回结果",
        }

        print(result)
        return result

    result = queue.get()

    print("测试结果:")
    print(result)

    return result


def main():
    print(
        "AKShare version:",
        getattr(ak, "__version__", "unknown")
    )

    sources = [
        # 同花顺：行业/概念板块概况
        "stock_board_industry_summary_ths",
        "stock_board_concept_summary_ths",

        # 同花顺：板块名称
        "stock_board_industry_name_ths",
        "stock_board_concept_name_ths",

        # 东财：行业实时板块
        "stock_board_industry_spot_em",
    ]

    results = []

    for source in sources:
        results.append(
            test_source(source)
        )

    print("")
    print("=" * 70)
    print("最终板块测速结果")
    print("=" * 70)

    for item in results:
        print(
            item.get("source"),
            "=>",
            item.get("status"),
            "|",
            item.get("seconds"),
            "秒",
            "| rows:",
            item.get("rows"),
            "| args:",
            item.get("required_params"),
        )


if __name__ == "__main__":
    main()
