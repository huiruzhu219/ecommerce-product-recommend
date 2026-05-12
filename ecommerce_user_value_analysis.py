from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(r"D:\archive\user_personalized_features.csv")
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def pct_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return a stable 0-100 percentile score."""
    ranks = series.rank(method="average", pct=True, ascending=True)
    if higher_is_better:
        return ranks * 100
    return (1 - ranks) * 100


def tier_by_quantile(series: pd.Series, labels: list[str]) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=len(labels), labels=labels)


def active_connection(row: pd.Series) -> str:
    subscribed = bool(row["Newsletter_Subscription"])
    days = row["Last_Login_Days_Ago"]
    if subscribed and days <= 7:
        return "高连接-订阅且7天内登录"
    if (not subscribed) and days <= 7:
        return "中连接-未订阅但7天内登录"
    if (subscribed and days >= 30) or days > 30:
        return "低连接-名义忠诚/不活跃"
    if subscribed:
        return "名义忠诚-需激活"
    return "弱连接-未订阅且近期未登录"


def segment_user(row: pd.Series) -> str:
    high_income = row["Income_Pct"] >= 75
    low_income = row["Income_Pct"] <= 40
    high_spend = row["M_Score"] >= 70
    low_spend = row["M_Score"] <= 45
    active_recent = row["Last_Login_Days_Ago"] <= 7
    inactive = row["R_Score"] <= 35
    high_intent = row["I_Score"] >= 70
    very_high_intent = row["I_Score"] >= 80
    high_friction = row["Friction_Score"] >= 70
    low_purchase = row["Purchase_Frequency"] <= row["Purchase_Frequency_Median"]

    if high_income and high_spend and inactive:
        return "高潜流失客"
    if high_income and high_intent and low_spend and high_friction:
        return "纠结土豪"
    if very_high_intent and active_recent and low_purchase and low_spend:
        return "高潜观望者"
    if high_intent and low_income and low_spend and row["Last_Login_Days_Ago"] <= 15:
        return "隐形活跃者"
    if row["Enhanced_Value_Score"] >= 78 and row["R_Score"] >= 45:
        return "核心用户"
    if row["Enhanced_Value_Score"] >= 62 or (high_intent and active_recent):
        return "潜力用户"
    if row["R_Score"] <= 30 and row["RFM_Score"] >= 55:
        return "沉睡价值用户"
    if row["RFM_Score"] <= 35 and row["I_Score"] <= 45:
        return "低价值用户"
    return "一般用户"


def roi_simulation(df: pd.DataFrame, coupon_value: float = 10.0, gross_margin: float = 0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proxy ROI experiment under equal budget.

    The dataset has no campaign exposure/outcome table, so this is a deterministic
    expected-value experiment using behavior-derived propensity and uplift proxies.
    """
    target_n = max(1, int(np.floor(len(df) * 0.20)))
    budget = target_n * coupon_value

    active_bonus = df["Active_Connection"].map(
        {
            "高连接-订阅且7天内登录": 0.35,
            "中连接-未订阅但7天内登录": 0.25,
            "名义忠诚-需激活": 0.05,
            "弱连接-未订阅且近期未登录": -0.10,
            "低连接-名义忠诚/不活跃": -0.25,
        }
    ).fillna(0)
    segment_bonus = df["Enhanced_Segment"].map(
        {
            "纠结土豪": 0.55,
            "高潜观望者": 0.45,
            "高潜流失客": 0.40,
            "潜力用户": 0.25,
            "核心用户": 0.15,
            "隐形活跃者": 0.10,
        }
    ).fillna(0)
    affordability = np.clip(df["Income_Pct"] / 100, 0.05, 1.0)
    friction_penalty = np.clip(1 - df["Friction_Score"] / 180, 0.35, 1)

    logit = (
        -3.2
        + 0.018 * df["I_Score"]
        + 0.010 * df["F_Score"]
        + 0.008 * df["M_Score"]
        + active_bonus
        + 0.25 * affordability
    )
    base_prob = 1 / (1 + np.exp(-logit))
    coupon_uplift = np.clip(
        0.035
        * (0.55 + df["I_Score"] / 100)
        * friction_penalty
        * (0.75 + affordability)
        + segment_bonus / 20,
        0.005,
        0.18,
    )
    expected_90d_value = df["Average_Order_Value"] * (1 + df["Purchase_Frequency"] / 5)
    expected_coupon_cost = coupon_value * (base_prob + coupon_uplift)
    expected_incremental_revenue = coupon_uplift * expected_90d_value
    expected_incremental_profit = expected_incremental_revenue * gross_margin - expected_coupon_cost

    exp = df[
        [
            "User_ID",
            "RFM_Score",
            "Enhanced_Value_Score",
            "Enhanced_Segment",
            "Average_Order_Value",
        ]
    ].copy()
    exp["Base_Conversion_Proxy"] = base_prob
    exp["Coupon_Uplift_Proxy"] = coupon_uplift
    exp["Expected_Coupon_Cost"] = expected_coupon_cost
    exp["Expected_Incremental_Revenue"] = expected_incremental_revenue
    exp["Expected_Incremental_Profit"] = expected_incremental_profit

    traditional = exp.nlargest(target_n, "RFM_Score").assign(Experiment_Group="传统RFM-Top20%")
    optimized_pool = exp[exp["Enhanced_Segment"].isin(["核心用户", "潜力用户", "纠结土豪", "高潜观望者", "高潜流失客"])]
    optimized = optimized_pool.nlargest(target_n, "Expected_Incremental_Profit").assign(Experiment_Group="优化RFM-核心与潜力")

    combined = pd.concat([traditional, optimized], ignore_index=True)
    summary = (
        combined.groupby("Experiment_Group")
        .agg(
            Target_Users=("User_ID", "count"),
            Budget=("User_ID", lambda s: len(s) * coupon_value),
            Avg_RFM=("RFM_Score", "mean"),
            Avg_Enhanced_Value=("Enhanced_Value_Score", "mean"),
            Avg_Base_Conversion=("Base_Conversion_Proxy", "mean"),
            Avg_Coupon_Uplift=("Coupon_Uplift_Proxy", "mean"),
            Expected_Coupon_Cost=("Expected_Coupon_Cost", "sum"),
            Expected_Incremental_Revenue=("Expected_Incremental_Revenue", "sum"),
            Expected_Incremental_Profit=("Expected_Incremental_Profit", "sum"),
        )
        .reset_index()
    )
    summary["Expected_ROI"] = summary["Expected_Incremental_Profit"] / summary["Budget"]
    summary["Budget"] = budget
    return combined, summary


SEGMENT_STRATEGY = {
    "核心用户": {
        "画像": "近期活跃、购买基础好、增强价值分最高，是平台当前最稳定的利润来源。",
        "主要阻力": "不是不买，而是可能被过度补贴；如果只给通用券，会浪费预算并训练其等券消费。",
        "增购打法": "用会员等级、专属新品、加价购、组合套装和复购周期提醒提升客单价与购买频次。",
        "推荐权益": "低折扣高感知权益，如会员日、积分加倍、满额赠、专属客服、预售优先权。",
        "商品/内容": "推荐高复购、高毛利、升级款、同类目套装和跨品类搭配。",
        "KPI": "复购率、客单价、毛利率、会员留存、权益使用率。",
    },
    "潜力用户": {
        "画像": "综合价值较高，购买和活跃表现尚可，但还没完全稳定为核心用户。",
        "主要阻力": "对平台信任或购买习惯仍在形成，可能需要更明确的利益点触发下一单。",
        "增购打法": "围绕最近偏好类目做精准推荐，用阶梯满减和连续购买奖励把偶发消费变成习惯消费。",
        "推荐权益": "首选限时满减、第二件优惠、连购奖励、品类券，不建议直接发大额无门槛券。",
        "商品/内容": "推荐浏览/购买类目的热销款、评价高商品、价格带略高的升级款。",
        "KPI": "30天复购率、品类渗透率、券后毛利、从潜力转核心的人数。",
    },
    "高潜流失客": {
        "画像": "收入高、历史消费高，但近期活跃下降，是最需要优先挽回的高价值人群。",
        "主要阻力": "可能出现价格敏感、体验不满、竞品迁移、需求周期结束或推荐不准。",
        "增购打法": "先做召回和诊断，再做个性化挽留；对高客单用户给专属服务比单纯降价更有效。",
        "推荐权益": "专属召回券、老客回归礼包、客服一对一、包邮/退换无忧、限时高价值权益。",
        "商品/内容": "围绕历史高消费类目推新品、升级款、补充耗材、同品牌系列。",
        "KPI": "召回率、回归首单金额、回归后60天留存、流失原因反馈率。",
    },
    "纠结土豪": {
        "画像": "收入高、意向强、浏览多但消费低，典型是有钱也想买，却没找到足够合适或足够确定的商品。",
        "主要阻力": "选择成本高、商品匹配不足、价格锚点不清、缺少确定性评价或服务保障。",
        "增购打法": "不要只砸大券，先缩短决策路径：个性化榜单、同价位对比、专家推荐、套装方案，再用小额限时券临门一脚。",
        "推荐权益": "高客单品类券、限时小额券、专属导购、试用/无忧退、品质保障。",
        "商品/内容": "推荐高收入匹配的品质款、品牌款、套装款，以及“帮我选”式内容。",
        "KPI": "浏览到加购转化率、加购到购买转化率、决策时长、客单价。",
    },
    "高潜观望者": {
        "画像": "天天逛、停留久、看很多页面，但购买少或不花钱，是新增模型专门识别的高意向未转化人群。",
        "主要阻力": "可能缺少首单信任、价格门槛、商品选择过载，或者只是把平台当信息查询工具。",
        "增购打法": "用低门槛首单、足迹召回、购物车提醒和爆品推荐促成首次/下一次购买；目标是先让其完成交易闭环。",
        "推荐权益": "首单券、低门槛品类券、限时免邮、浏览商品降价提醒。",
        "商品/内容": "推荐其高频浏览类目中的爆品、低决策成本商品、评价高且价格适中的商品。",
        "KPI": "首单转化率、浏览后购买率、券核销率、从观望转潜力的人数。",
    },
    "隐形活跃者": {
        "画像": "意向高、活跃或半活跃，但收入和消费较低，可能更像平台内容粉丝或价格敏感型用户。",
        "主要阻力": "购买力有限，对高价商品承接弱；强推高价会降低体验。",
        "增购打法": "让他们先带来低成本价值：分享裂变、评价晒单、内容互动；消费侧用低价爆品和拼团提升频次。",
        "推荐权益": "分享返利、邀请奖励、积分兑换、拼团券、低门槛小额券。",
        "商品/内容": "推荐平价爆品、试用装、小规格、组合优惠和社交传播素材。",
        "KPI": "分享人数、邀请新客、低价品复购、内容互动率、裂变 ROI。",
    },
    "沉睡价值用户": {
        "画像": "历史价值不低但近期沉睡，仍有唤醒可能，但优先级低于高潜流失客。",
        "主要阻力": "需求弱化、平台记忆下降、触达疲劳或替代平台迁移。",
        "增购打法": "用轻触达测试兴趣，再根据点击/回访行为分层加码，避免一开始就高补贴。",
        "推荐权益": "回访提醒、兴趣测试券、老客专属小礼包、曾购类目补货提醒。",
        "商品/内容": "历史购买类目的新品、消耗品补货、季节性商品。",
        "KPI": "唤醒点击率、回访率、唤醒后购买率、触达退订率。",
    },
    "一般用户": {
        "画像": "各项指标居中或特征不明显，是规模最大的人群，适合自动化运营和二次细分。",
        "主要阻力": "需求、意向、消费能力和活跃度都不突出，单独大额补贴效率低。",
        "增购打法": "用算法推荐和生命周期任务培养偏好，观察其向潜力、观望或流失方向迁移。",
        "推荐权益": "普通品类券、签到积分、浏览后推荐、常规促销。",
        "商品/内容": "平台热销、个性化猜你喜欢、低风险试购商品。",
        "KPI": "活跃率、偏好识别率、加购率、向潜力用户迁移率。",
    },
    "低价值用户": {
        "画像": "消费低、意向低或近期不活跃，短期直接拉动消费的性价比较低。",
        "主要阻力": "需求弱、平台关系弱、价格敏感或并非目标客群。",
        "增购打法": "控制预算，主要用自动化低成本触达；只有出现高意向行为后再升级运营。",
        "推荐权益": "签到、积分、小额低门槛券、清仓品类券。",
        "商品/内容": "低价爆品、清仓商品、兴趣测试内容。",
        "KPI": "低成本唤醒率、触达成本、是否向一般/潜力用户迁移。",
    },
}


def build_strategy_table(segment_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment, metrics in segment_summary.iterrows():
        strategy = SEGMENT_STRATEGY.get(segment, {})
        rows.append(
            {
                "用户分层": segment,
                "用户数": int(metrics["用户数"]),
                "平均消费": round(metrics["平均消费"], 2),
                "平均收入": round(metrics["平均收入"], 2),
                "平均意向分": round(metrics["平均意向分"], 2),
                "平均摩擦分": round(metrics["平均摩擦分"], 2),
                "用户画像": strategy.get("画像", ""),
                "消费阻力": strategy.get("主要阻力", ""),
                "让他们多花钱的策略": strategy.get("增购打法", ""),
                "推荐权益/优惠": strategy.get("推荐权益", ""),
                "推荐商品或内容": strategy.get("商品/内容", ""),
                "运营KPI": strategy.get("KPI", ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed|^$", regex=True)]

    bool_map = {"True": True, "False": False, True: True, False: False}
    df["Newsletter_Subscription"] = df["Newsletter_Subscription"].map(bool_map).astype(bool)
    numeric_cols = [
        "Age",
        "Income",
        "Last_Login_Days_Ago",
        "Purchase_Frequency",
        "Average_Order_Value",
        "Total_Spending",
        "Time_Spent_on_Site_Minutes",
        "Pages_Viewed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    df["R_Score"] = pct_score(df["Last_Login_Days_Ago"], higher_is_better=False)
    df["F_Score"] = pct_score(df["Purchase_Frequency"], higher_is_better=True)
    df["M_Score"] = pct_score(df["Total_Spending"], higher_is_better=True)
    df["RFM_Score"] = 0.35 * df["R_Score"] + 0.30 * df["F_Score"] + 0.35 * df["M_Score"]

    df["Time_Score"] = pct_score(df["Time_Spent_on_Site_Minutes"], higher_is_better=True)
    df["Pages_Score"] = pct_score(df["Pages_Viewed"], higher_is_better=True)
    df["I_Score"] = 0.5 * df["Time_Score"] + 0.5 * df["Pages_Score"]
    df["Conversion_Friction"] = df["Pages_Viewed"] / (df["Purchase_Frequency"] + 1)
    df["Friction_Score"] = pct_score(df["Conversion_Friction"], higher_is_better=True)
    df["Active_Connection"] = df.apply(active_connection, axis=1)

    df["Income_Pct"] = pct_score(df["Income"], higher_is_better=True)
    df["Income_Tier"] = tier_by_quantile(df["Income"], ["低收入", "中低收入", "中高收入", "高收入"])
    df["Spend_Income_Ratio"] = df["Total_Spending"] / df["Income"]
    df["Spend_Burden_Tier"] = tier_by_quantile(
        df["Spend_Income_Ratio"],
        ["低负担消费", "适中负担消费", "高负担消费", "超高负担消费"],
    )
    df["Purchasing_Power_Tag"] = df["Income_Tier"].astype(str) + "/" + df["Spend_Burden_Tier"].astype(str)

    active_adj = df["Active_Connection"].map(
        {
            "高连接-订阅且7天内登录": 8,
            "中连接-未订阅但7天内登录": 5,
            "名义忠诚-需激活": 1,
            "弱连接-未订阅且近期未登录": -3,
            "低连接-名义忠诚/不活跃": -8,
        }
    ).fillna(0)
    friction_adj = np.where(df["Friction_Score"] >= 75, -5, np.where(df["Friction_Score"] <= 25, 4, 0))
    df["Enhanced_Value_Score"] = np.clip(
        0.55 * df["RFM_Score"] + 0.25 * df["I_Score"] + 0.20 * (100 - df["Friction_Score"]) + active_adj + friction_adj,
        0,
        100,
    )

    df["Purchase_Frequency_Median"] = df["Purchase_Frequency"].median()
    df["Enhanced_Segment"] = df.apply(segment_user, axis=1)
    df = df.drop(columns=["Purchase_Frequency_Median"])

    user_cols = [
        "User_ID",
        "Income",
        "Income_Tier",
        "Spend_Income_Ratio",
        "Purchasing_Power_Tag",
        "Last_Login_Days_Ago",
        "Purchase_Frequency",
        "Average_Order_Value",
        "Total_Spending",
        "Time_Spent_on_Site_Minutes",
        "Pages_Viewed",
        "Newsletter_Subscription",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score",
        "I_Score",
        "Conversion_Friction",
        "Friction_Score",
        "Active_Connection",
        "Enhanced_Value_Score",
        "Enhanced_Segment",
        "Product_Category_Preference",
        "Interests",
    ]
    df[user_cols].to_csv(OUT_DIR / "user_value_segments.csv", index=False, encoding="utf-8-sig")

    segment_summary = (
        df.groupby("Enhanced_Segment")
        .agg(
            用户数=("User_ID", "count"),
            平均消费=("Total_Spending", "mean"),
            平均客单价=("Average_Order_Value", "mean"),
            平均收入=("Income", "mean"),
            平均意向分=("I_Score", "mean"),
            平均摩擦分=("Friction_Score", "mean"),
            平均增强价值分=("Enhanced_Value_Score", "mean"),
            最近登录天数=("Last_Login_Days_Ago", "mean"),
        )
        .sort_values("用户数", ascending=False)
        .round(2)
    )
    segment_summary.to_csv(OUT_DIR / "segment_summary.csv", encoding="utf-8-sig")
    strategy_table = build_strategy_table(segment_summary)
    strategy_table.to_csv(OUT_DIR / "segment_marketing_strategy.csv", index=False, encoding="utf-8-sig")

    roi_detail, roi_summary = roi_simulation(df)
    roi_detail.to_csv(OUT_DIR / "roi_experiment_detail.csv", index=False, encoding="utf-8-sig")
    roi_summary.round(4).to_csv(OUT_DIR / "roi_experiment_summary.csv", index=False, encoding="utf-8-sig")

    report = build_report(df, segment_summary, roi_summary, strategy_table)
    (OUT_DIR / "ecommerce_user_value_report.md").write_text(report, encoding="utf-8")


def build_report(
    df: pd.DataFrame,
    segment_summary: pd.DataFrame,
    roi_summary: pd.DataFrame,
    strategy_table: pd.DataFrame,
) -> str:
    seg_lines = []
    for seg, row in segment_summary.iterrows():
        seg_lines.append(
            f"| {seg} | {int(row['用户数'])} | {row['平均消费']:.2f} | {row['平均收入']:.2f} | "
            f"{row['平均意向分']:.2f} | {row['平均摩擦分']:.2f} | {row['平均增强价值分']:.2f} | {row['最近登录天数']:.2f} |"
        )

    roi_lines = []
    for _, row in roi_summary.round(4).iterrows():
        roi_lines.append(
            f"| {row['Experiment_Group']} | {int(row['Target_Users'])} | {row['Budget']:.0f} | "
            f"{row['Avg_RFM']:.2f} | {row['Avg_Enhanced_Value']:.2f} | {row['Avg_Coupon_Uplift']:.4f} | "
            f"{row['Expected_Coupon_Cost']:.2f} | {row['Expected_Incremental_Revenue']:.2f} | "
            f"{row['Expected_Incremental_Profit']:.2f} | {row['Expected_ROI']:.4f} |"
        )

    strategy_lines = []
    for row in strategy_table.to_dict("records"):
        strategy_lines.append(
            f"### {row['用户分层']}\n\n"
            f"- 用户画像：{row['用户画像']}\n"
            f"- 消费阻力：{row['消费阻力']}\n"
            f"- 让他们多花钱的策略：{row['让他们多花钱的策略']}\n"
            f"- 推荐权益/优惠：{row['推荐权益/优惠']}\n"
            f"- 推荐商品或内容：{row['推荐商品或内容']}\n"
            f"- 运营 KPI：{row['运营KPI']}\n"
        )

    top_segments = df["Enhanced_Segment"].value_counts()
    key_counts = ", ".join([f"{k}{v}人" for k, v in top_segments.items()])
    high_observers = df[df["Enhanced_Segment"].eq("高潜观望者")]
    burden_example = (
        df.assign(Spend_Income_Ratio_Pct=df["Spend_Income_Ratio"] * 100)
        .sort_values(["Total_Spending", "Spend_Income_Ratio"], ascending=[False, False])
        .head(3)[["User_ID", "Income", "Total_Spending", "Spend_Income_Ratio_Pct", "Purchasing_Power_Tag"]]
    )
    burden_lines = [
        f"- {r.User_ID}: 收入 {r.Income:.0f}, 消费 {r.Total_Spending:.0f}, 消费/收入 {r.Spend_Income_Ratio_Pct:.2f}%, 标签 {r.Purchasing_Power_Tag}"
        for r in burden_example.itertuples()
    ]

    return f"""# 电商用户价值分层与精准营销策略报告

## 1. 数据与建模口径

- 样本量：{len(df)} 名用户。
- 传统 RFM：R=最近登录越近越高，F=购买次数越多越高，M=总消费越高越高；综合分 = 0.35R + 0.30F + 0.35M。
- 新增意向度：I-score = 停留时长分位分 * 0.5 + 浏览页数分位分 * 0.5，用于衡量“想不想买”。
- 新增转化摩擦系数：浏览页数 / (购买次数 + 1)，越高代表看得多、买得少，决策更纠结。
- 新增活跃连接度：订阅且 7 天内登录为高连接；未订阅但 7 天内登录为中连接；订阅但 30 天未登录或长期不活跃为低连接；中间状态单独标记为需激活或弱连接。
- 购买力背景不计入综合价值分，只作为修正标签：收入分层 + 消费/收入比例。这样能区分“月薪五万花 1000”和“月薪五千花 1000”的不同购买压力。

增强价值分 = 0.55 * RFM + 0.25 * I-score + 0.20 * (100 - 摩擦分) + 活跃连接修正 + 高/低摩擦修正。

## 2. 更新后用户分层结果

分层规模：{key_counts}。

| 用户分层 | 用户数 | 平均消费 | 平均收入 | 平均意向分 | 平均摩擦分 | 平均增强价值分 | 最近登录天数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seg_lines)}

## 3. 重点人群解释与策略

- 纠结土豪：高收入、高意向、低消费且摩擦高。重点不是大水漫灌优惠，而是商品匹配、价格锚点、套装推荐和有限时效优惠，降低“看了很多但没找到合适商品”的决策成本。
- 隐形活跃者：高意向、低收入、低消费且仍保持活跃。更适合会员成长、内容种草、分享返利、裂变奖励，不宜直接推高价商品。
- 高潜流失客：高收入、高消费但近期活跃下降。需要优先触达，询问流失原因，提供专属客服、个性化挽留券或新品优先体验。
- 高潜观望者：本次新增识别，共 {len(high_observers)} 人。典型表现是停留久、浏览多、近期活跃，但购买少或消费低。策略应以首单/低门槛券、浏览品类的精准推荐、购物车/足迹召回为主。
- 核心用户：增强价值高且近期仍有活跃。适合权益升级、复购券、会员专属推荐，但优惠力度不宜过高，避免补贴本来就会购买的人。

## 4. 分层用户画像与增购策略

{chr(10).join(strategy_lines)}

## 5. 购买力背景修正

购买力不进入主分，避免把“有钱但没需求”误判为高价值；但在运营策略中必须作为标签参与优惠力度和商品价格带选择。

示例：
{chr(10).join(burden_lines)}

建议使用：
- 高收入/低负担消费：可推高客单、品质升级、套装组合，用服务和品质理由让他们花更多，而不是单纯便宜。
- 高收入/高摩擦：先解决选择障碍，再给小额限时激励，重点提高转化率和客单价。
- 低收入/高负担消费：说明消费投入相对重，更适合分期、满减门槛下调、平价替代品和忠诚权益，不能盲目推高价。
- 低收入/低负担消费：不要强推高价商品，优先做内容互动、分享返利和低门槛转化。

## 6. ROI 对比实验设计

实验目标：在相同优惠券预算下，比较传统 RFM 与优化 RFM 的营销 ROI。

实验设置：
- 预算固定：传统策略投放前 20% 高 RFM 用户，共 {int(len(df) * 0.2)} 人；单用户券面预算设为 10。
- 传统组：按 RFM_Score 排名前 20% 发券。
- 优化组：只在核心用户、潜力用户、纠结土豪、高潜观望者、高潜流失客中选择，并按预期增量利润排序，人数与传统组一致。
- 由于原始数据没有真实营销曝光和转化结果，本报告采用行为代理模型估算预期 ROI；真实上线时应替换为 A/B 实验实际订单毛利、券核销和增量购买数据。

| 实验组 | 触达用户 | 预算 | 平均RFM | 平均增强价值分 | 平均券增量率 | 预期核销成本 | 预期增量销售额 | 预期增量利润 | 预期ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(roi_lines)}

## 7. 结论

传统 RFM 更擅长找到“已经花过钱的人”，但会漏掉高意向低消费、浏览多但未转化、以及高消费但正在流失的人。优化模型通过 I-score、转化摩擦、活跃连接度和购买力背景标签，把优惠券从单纯奖励历史消费，改成优先投向更可能产生增量的人群。

落地优先级：先挽回高潜流失客，再转化纠结土豪和高潜观望者，最后用低成本权益经营隐形活跃者与核心用户。
"""


if __name__ == "__main__":
    main()
