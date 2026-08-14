import time
import akshare as ak


TEST_STOCKS = [
    ("002349", "精华制药"),
    ("002033", "丽江股份"),
]


def main():
    print("测试巨潮资讯个股行业归属")
    print("=" * 70)

    for code, name in TEST_STOCKS:

        print("")
        print("=" * 70)
        print(f"开始测试: {code} {name}")
        print("=" * 70)

        start = time.time()

        try:
            df = ak.stock_industry_change_cninfo(
                symbol=code,
                start_date="20000101",
                end_date="20260814"
            )

            elapsed = time.time() - start

            print(
                f"{code} {name} => success | "
                f"{elapsed:.2f} 秒 | "
                f"rows: {len(df)}"
            )

            print("")
            print("columns:")
            print(list(df.columns))

            print("")
            print("全部分类标准:")
            if "分类标准" in df.columns:
                print(
                    df["分类标准"]
                    .dropna()
                    .unique()
                    .tolist()
                )

            print("")
            print("最新记录:")
            print(
                df.tail(20).to_dict("records")
            )

        except Exception as e:

            elapsed = time.time() - start

            print(
                f"{code} {name} => error | "
                f"{elapsed:.2f} 秒"
            )

            print(repr(e))


if __name__ == "__main__":
    main()
