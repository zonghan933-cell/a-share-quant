import time
import akshare as ak


TEST_INDUSTRIES = [
    "电子化学品",
    "通信设备",
    "元件",
]


def main():
    print("测试同花顺行业成分股接口")
    print("=" * 60)

    for industry in TEST_INDUSTRIES:
        print("")
        print("开始测试:", industry)

        start = time.time()

        try:
            df = ak.stock_board_industry_cons_ths(
                symbol=industry
            )

            elapsed = time.time() - start

            print(
                f"{industry} => success | "
                f"{elapsed:.2f} 秒 | "
                f"rows: {len(df)}"
            )

            print("columns:")
            print(list(df.columns))

            print("sample:")
            print(
                df.head(3).to_dict("records")
            )

        except Exception as e:
            elapsed = time.time() - start

            print(
                f"{industry} => error | "
                f"{elapsed:.2f} 秒"
            )

            print(repr(e))


if __name__ == "__main__":
    main()
