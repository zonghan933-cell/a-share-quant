import time
import akshare as ak


def show_result(title, df, elapsed):
    print("")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"耗时: {elapsed:.2f} 秒")
    print(f"rows: {len(df)}")

    print("")
    print("columns:")
    print(list(df.columns))

    print("")
    print("sample:")
    print(df.head(8).to_dict("records"))


def main():
    print("A股行业映射 + 行业资金流测试")
    print("=" * 70)

    # ==========================================
    # 1. 申万全市场股票行业分类
    # ==========================================

    print("")
    print("开始测试: stock_industry_clf_hist_sw")

    start = time.time()

    try:
        clf = ak.stock_industry_clf_hist_sw()

        elapsed = time.time() - start

        show_result(
            "申万个股行业分类",
            clf,
            elapsed
        )

        # 看看哪些字段与“行业”有关
        industry_cols = [
            col
            for col in clf.columns
            if "行业" in str(col)
            or "industry" in str(col).lower()
        ]

        print("")
        print("可能的行业字段:")
        print(industry_cols)

        # 如果有股票代码字段，取最新行业记录
        if "symbol" in clf.columns:
            current_clf = clf.copy()

            if "start_date" in current_clf.columns:
                current_clf = current_clf.sort_values(
                    "start_date"
                )

            current_clf = current_clf.drop_duplicates(
                subset=["symbol"],
                keep="last"
            )

            print("")
            print(
                "去重后当前股票数量:",
                len(current_clf)
            )

            print("当前分类 sample:")
            print(
                current_clf.head(10).to_dict(
                    "records"
                )
            )

    except Exception as e:
        print("")
        print(
            "stock_industry_clf_hist_sw => error"
        )
        print(repr(e))

    # ==========================================
    # 2. 同花顺即时行业资金流
    # ==========================================

    print("")
    print("开始测试: stock_fund_flow_industry 即时")

    start = time.time()

    try:
        fund_now = ak.stock_fund_flow_industry(
            symbol="即时"
        )

        elapsed = time.time() - start

        show_result(
            "同花顺即时行业资金流",
            fund_now,
            elapsed
        )

    except Exception as e:
        print("")
        print(
            "stock_fund_flow_industry 即时 => error"
        )
        print(repr(e))

    # ==========================================
    # 3. 同花顺3日行业资金流
    # ==========================================

    print("")
    print("开始测试: stock_fund_flow_industry 3日排行")

    start = time.time()

    try:
        fund_3d = ak.stock_fund_flow_industry(
            symbol="3日排行"
        )

        elapsed = time.time() - start

        show_result(
            "同花顺3日行业资金流",
            fund_3d,
            elapsed
        )

    except Exception as e:
        print("")
        print(
            "stock_fund_flow_industry 3日 => error"
        )
        print(repr(e))


if __name__ == "__main__":
    main()
