import inspect
import multiprocessing as mp
import time
import traceback

import akshare as ak


TIMEOUT_SECONDS = 75


def worker(func_name, queue):
    start = time.time()

    try:
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
            queue.put(
                {
                    "source": func_name,
                    "status": "skip",
                    "reason": f"需要必填参数: {required_params}",
                    "seconds": round(time.time() - start, 2),
                }
            )
            return

        df = func()

        elapsed = round(time.time() - start, 2)

        if df is None:
            queue.put(
                {
                    "source": func_name,
                    "status": "error",
                    "reason": "返回 None",
                    "seconds": elapsed,
                }
            )
            return

        queue.put(
            {
                "source": func_name,
                "status": "success",
                "rows": len(df),
                "columns": list(df.columns),
                "seconds": elapsed,
                "sample": df.head(2).to_dict("records"),
            }
        )

    except Exception as e:
        queue.put(
            {
                "source": func_name,
                "status": "error",
                "reason": repr(e),
                "traceback": traceback.format_exc()[-2000:],
                "seconds": round(time.time() - start, 2),
            }
        )


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
            "reason": "子进程退出但没有返回结果",
        }

        print(result)
        return result

    result = queue.get()

    print("测试结果:")
    print(result)

    return result


def main():
    print("AKShare version:", getattr(ak, "__version__", "unknown"))

    sources = [
        "stock_zh_a_spot_em",   # 东方财富
        "stock_zh_a_spot_tx",   # 腾讯
        "stock_zh_a_spot",      # 新浪
    ]

    results = []

    for source in sources:
        results.append(test_source(source))

    print("")
    print("=" * 70)
    print("最终测速结果")
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
        )

    successful = [
        x
        for x in results
        if x.get("status") == "success"
    ]

    if successful:
        fastest = min(
            successful,
            key=lambda x: x.get("seconds", 999999)
        )

        print("")
        print("推荐数据源:")
        print(fastest["source"])
        print(
            f"耗时: {fastest['seconds']} 秒, "
            f"返回 {fastest['rows']} 条"
        )
    else:
        print("")
        print("三个数据源都没有在限定时间内成功。")


if __name__ == "__main__":
    main()
