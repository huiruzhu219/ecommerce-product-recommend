from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs_jupyter"
IMG_DIR = BASE / "ecommerce_rfm_analysis_notebook_pycharm_fixed_files"
REPORT = BASE / "电商用户价值分层与精准营销策略_图文报告.md"
HTML_REPORT = BASE / "电商用户价值分层与精准营销策略_图文报告.html"


def md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns]
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.2f}")
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join([header, sep, *rows])


def img(name: str, caption: str) -> str:
    path = IMG_DIR / name
    rel = path.relative_to(BASE).as_posix()
    return f"\n![{caption}]({rel})\n\n*{caption}*\n"


def main() -> None:
    user = pd.read_csv(OUT / "user_value_segments_jupyter.csv")
    segment = pd.read_csv(OUT / "segment_summary_jupyter.csv")
    strategy = pd.read_csv(OUT / "segment_marketing_strategy_jupyter.csv")
    roi = pd.read_csv(OUT / "roi_experiment_summary_jupyter.csv")

    opt = roi[roi["Experiment_Group"].str.contains("优化")].iloc[0]
    trad = roi[roi["Experiment_Group"].str.contains("传统")].iloc[0]
    roi_lift = opt["Expected_ROI"] / trad["Expected_ROI"] if trad["Expected_ROI"] else float("inf")
    revenue_lift = opt["Expected_Incremental_Revenue"] - trad["Expected_Incremental_Revenue"]
    profit_lift = opt["Expected_Incremental_Profit"] - trad["Expected_Incremental_Profit"]

    top_counts = "、".join(
        f"{r.Enhanced_Segment}{int(r.用户数)}人" for r in segment.itertuples(index=False)
    )

    key_strategy = strategy[strategy["用户分层"].isin(["核心用户", "潜力用户", "高潜流失客", "纠结土豪", "高潜观望者", "隐形活跃者"])]
    segment_view = segment.rename(columns={"Enhanced_Segment": "用户分层"})
    roi_view = roi.rename(
        columns={
            "Experiment_Group": "实验组",
            "Target_Users": "触达用户数",
            "Budget": "券面预算",
            "Avg_RFM": "平均RFM",
            "Avg_Enhanced_Value": "平均增强价值分",
            "Avg_Coupon_Uplift": "平均券增量率",
            "Expected_Coupon_Cost": "预期核销成本",
            "Expected_Incremental_Revenue": "预期增量销售额",
            "Expected_Incremental_Profit": "预期增量利润",
            "Expected_ROI": "预期ROI",
        }
    )

    report = f"""# 电商用户价值分层与精准营销策略图文报告

## 一、项目背景

在电商运营中，传统用户价值评估常依赖 RFM 模型，即最近一次行为、购买频次和消费金额。该方法能够快速找到历史消费能力较强的用户，但它主要解释“过去谁买得多”，对“现在谁更想买”“谁看了很多却没买”“谁正在流失”“谁有购买力但缺少合适商品”等问题识别不足。

本项目基于 `user_personalized_features.csv` 共 {len(user)} 名用户数据，在传统 RFM 的基础上引入意向度、转化摩擦、活跃连接度和购买力背景标签，构建优化后的 RFM-I 用户价值分层模型，并进一步提出分层营销策略和 ROI 对比实验。

本报告的核心目标是：把用户从简单的“高价值/低价值”划分，升级为可运营、可触达、可转化的用户画像体系。

## 二、传统 RFM 策略的不足

传统 RFM 策略通常把优惠券投放给 RFM 排名前 20% 的用户。这种做法简单直接，但存在明显问题：

1. 容易补贴本来就会购买的用户，导致营销预算浪费。
2. 难以识别“天天逛、停留久、浏览多，但就是不买”的高潜观望者。
3. 难以识别“收入高、意向强、但选择成本高”的纠结土豪。
4. 难以区分行为忠诚与名义忠诚，例如订阅了但长期不登录的用户。
5. 无法体现购买力差异，同样消费 1000 元，对高收入用户和低收入用户的含义完全不同。

因此，本项目不是否定 RFM，而是在 RFM 的基础上补充行为意向、转化阻力和购买力背景，让模型更接近真实运营场景。

## 三、优化 RFM-I 模型设计

### 1. 传统 RFM 维度

- R-score：最近登录越近，得分越高。
- F-score：购买次数越多，得分越高。
- M-score：总消费金额越高，得分越高。
- RFM 综合分 = 0.35 × R + 0.30 × F + 0.35 × M。

### 2. 新增修正因子

- 意向度 I-score = 停留时长分位分 × 0.5 + 浏览页数分位分 × 0.5，用于衡量用户“想不想买”。
- 转化摩擦系数 = 浏览页数 /（购买次数 + 1），用于衡量用户“看了多少页才买一次”，分值越高说明越纠结。
- 活跃连接度：结合订阅状态与最近登录天数，区分高连接、中连接、名义忠诚、弱连接和低连接用户。
- 购买力背景：基于收入分层和消费/收入比例形成标签，不进入综合得分，但用于修正营销动作。

### 3. 增强价值分

增强价值分 = 0.55 × RFM + 0.25 × I-score + 0.20 ×（100 - 摩擦分）+ 活跃连接修正 + 高/低摩擦修正。

该设计的含义是：历史价值仍然重要，但用户当前意向、转化阻力和活跃连接状态也会显著影响营销优先级。

## 四、用户分层结果

更新模型共识别出以下用户群体：{top_counts}。

{md_table(segment_view, ["用户分层", "用户数", "平均消费", "平均收入", "平均意向分", "平均摩擦分", "平均增强价值分", "最近登录天数"])}

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_10_0.png", "图1 更新 RFM-I 模型用户分层规模")}

从分层规模看，一般用户仍然占比最高，说明大部分用户特征尚不突出，需要通过自动化运营继续观察其迁移方向。潜力用户达到 {int(segment.loc[segment["Enhanced_Segment"].eq("潜力用户"), "用户数"].iloc[0])} 人，是后续转化和培养的主要对象。核心用户、高潜流失客、纠结土豪、高潜观望者虽然人数较少，但运营价值更集中，适合采用更精细的触达策略。

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_12_0.png", "图2 不同用户分层的核心指标对比")}

图2 展示了不同分层在消费、收入、意向、摩擦和增强价值上的差异。高潜流失客平均消费和平均收入较高，但最近登录天数偏长，说明其价值高但存在流失风险。纠结土豪收入和意向都高，但消费偏低、摩擦分很高，说明他们不是没钱，而是决策链路没有被打通。高潜观望者意向和摩擦都很高，是传统 RFM 最容易漏掉的新增重点人群。

## 五、关键可视化洞察

### 1. 意向度与消费金额

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_14_0.png", "图3 用户意向度 vs 总消费")}

图3 用意向度和总消费金额交叉观察用户。右下区域代表高意向但低消费用户，这类用户对平台有明显兴趣，却没有形成足够消费，是首单券、品类券、足迹召回和精准推荐的重点对象。右上区域则代表高意向高消费用户，适合做会员权益和复购经营。

### 2. 意向度与转化摩擦

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_16_0.png", "图4 意向度 vs 转化摩擦")}

图4 右上角是高意向、高摩擦用户，典型包括纠结土豪和高潜观望者。这类用户已经表现出较强购买意愿，但购买路径中存在选择障碍、价格犹豫、商品匹配不足或信任不足。对他们的营销重点不是简单发大额优惠券，而是降低决策成本，例如商品对比、榜单推荐、专家导购、套装方案、降价提醒和小额限时激励。

### 3. 传统 RFM 与增强价值分

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_18_0.png", "图5 传统 RFM 分数 vs 增强价值分")}

图5 展示了传统 RFM 与增强价值分之间的差异。部分用户传统 RFM 分不高，但增强价值分明显更高，说明他们可能具有当前意向、活跃连接或转化潜力。优化模型的价值正在于把这些传统模型低估的人群识别出来。

### 4. 购买力背景差异

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_20_0.png", "图6 收入 vs 消费金额")}

图6 说明同样的消费金额对不同收入用户意义不同。高收入用户花费 1000 元可能只是低负担消费，而低收入用户花费 1000 元可能已经是较高投入。因此，收入不应直接进入主价值分，否则容易把“有钱但无需求”的用户误判为高价值；但收入必须作为运营修正标签，用于决定商品价格带、优惠力度和沟通方式。

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_21_0.png", "图7 收入分层 × 消费负担热力图")}

图7 将收入分层与消费负担结合，帮助运营团队区分不同购买压力下的用户。高收入低负担用户适合推高客单和品质升级；低收入高负担用户更适合低门槛优惠、分期、平价替代品和忠诚权益。

## 六、分层用户画像与精准营销策略

{md_table(key_strategy, ["用户分层", "用户数", "平均消费", "平均收入", "平均意向分", "平均摩擦分", "营销建议"])}

### 1. 核心用户

核心用户增强价值分最高，近期活跃度较好，是平台当前最稳定的利润来源。运营重点不是盲目给大额券，而是通过会员等级、专属新品、加价购、组合套装和复购提醒提升客单价与复购频次。对这类用户，低折扣但高感知的权益往往比直接降价更有效。

### 2. 潜力用户

潜力用户是从普通用户向核心用户迁移的关键群体。其消费和活跃表现已经具备基础，但购买习惯尚未完全稳定。应围绕其最近浏览或购买偏好做精准推荐，并使用阶梯满减、第二件优惠、连购奖励等方式，把偶发消费转化为稳定复购。

### 3. 高潜流失客

高潜流失客收入高、历史消费高，但近期活跃下降。该群体如果流失，损失较大，应优先触达。建议先进行流失原因诊断，再提供专属召回券、客服一对一、老客回归礼包、新品优先体验等挽留方案。

### 4. 纠结土豪

纠结土豪有钱、有意向，但消费低、摩擦高。说明他们并非没有购买力，而是没有找到合适商品或缺少足够决策信心。运营应先提供高匹配商品推荐、同价位对比、专家推荐、品质保障和套装方案，再用小额限时券推动成交。

### 5. 高潜观望者

高潜观望者是新增模型重点识别的人群。他们经常浏览、停留时间长、页面浏览多，但购买少。对这类用户，目标应是先完成交易闭环，可以使用首单券、低门槛品类券、浏览商品降价提醒、购物车召回和爆品推荐。

### 6. 隐形活跃者

隐形活跃者意向高但收入和消费较低，可能更像内容粉丝或价格敏感型用户。对他们不适合强推高价商品，应鼓励分享裂变、评价晒单、内容互动、拼团和低价爆品复购，让其贡献传播价值和低成本订单。

## 七、ROI 对比实验

实验设计如下：

- 传统 RFM 策略：将优惠券发给 RFM_Score 前 20% 用户。
- 优化 RFM-I 策略：将优惠券发给核心用户、潜力用户、纠结土豪、高潜观望者和高潜流失客。
- 两组触达人数均为 200 人，券面预算均为 2000。
- 原始数据没有真实营销曝光和转化结果，因此本报告使用行为代理模型估算预期 ROI，正式上线时应使用 A/B 实验真实数据替代。

{md_table(roi_view, ["实验组", "触达用户数", "券面预算", "平均RFM", "平均增强价值分", "平均券增量率", "预期核销成本", "预期增量销售额", "预期增量利润", "预期ROI"])}

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_27_0.png", "图8 传统 RFM 与优化 RFM-I 的 ROI 对比")}

从 ROI 结果看，优化 RFM-I 策略在同等预算下的预期增量销售额为 {opt["Expected_Incremental_Revenue"]:.2f}，高于传统 RFM 的 {trad["Expected_Incremental_Revenue"]:.2f}，增量差额为 {revenue_lift:.2f}。预期增量利润从传统策略的 {trad["Expected_Incremental_Profit"]:.2f} 提升到 {opt["Expected_Incremental_Profit"]:.2f}，提升 {profit_lift:.2f}。预期 ROI 从 {trad["Expected_ROI"]:.4f} 提升到 {opt["Expected_ROI"]:.4f}，约为传统策略的 {roi_lift:.1f} 倍。

{img("ecommerce_rfm_analysis_notebook_pycharm_fixed_29_0.png", "图9 传统策略与优化策略触达人群结构对比")}

图9 表明，两种策略触达的人群结构明显不同。传统 RFM 更偏向历史消费和综合 RFM 排名靠前的人，而优化策略会把预算更多分配给核心、潜力、纠结、高潜观望和高潜流失等更可能产生增量的人群。这正是优化模型相较传统模型的主要价值。

## 八、最终结论

1. 传统 RFM 能够识别历史高价值用户，但容易忽略高意向未转化、浏览纠结、活跃下降和购买力背景差异。
2. 优化 RFM-I 模型通过意向度、转化摩擦、活跃连接度和购买力背景标签，让用户分层更接近真实运营场景。
3. 纠结土豪、高潜观望者、高潜流失客是传统模型不容易看见但极具运营价值的人群。
4. 营销策略应从“给高价值用户发券”升级为“针对不同用户阻力设计不同转化动作”。
5. ROI 对比显示，在相同预算下，优化 RFM-I 策略比传统 Top20% RFM 投放更能带来预期增量收益。

## 九、落地建议

- 第一优先级：高潜流失客。及时召回，避免高价值用户流失。
- 第二优先级：纠结土豪与高潜观望者。通过精准推荐和决策辅助促成转化。
- 第三优先级：核心用户与潜力用户。用会员权益、复购提醒和组合套装提升客单价。
- 第四优先级：隐形活跃者。以分享裂变、低价爆品和低成本互动为主。
- 长期机制：持续追踪用户在不同分层之间的迁移，并用真实 A/B 实验数据校准 ROI 模型。

"""

    REPORT.write_text(report, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>电商用户价值分层与精准营销策略图文报告</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; max-width: 1080px; margin: 32px auto; line-height: 1.75; color: #1f2937; }}
    h1, h2, h3 {{ color: #17324d; }}
    h1 {{ border-bottom: 4px solid #2f80ed; padding-bottom: 12px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8e2ef; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #eef5ff; }}
    img {{ max-width: 100%; display: block; margin: 20px auto 8px; border: 1px solid #e5e7eb; }}
    em {{ display: block; text-align: center; color: #607085; margin-bottom: 24px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
{report_to_html(report)}
</body>
</html>"""
    HTML_REPORT.write_text(html, encoding="utf-8")
    print(REPORT)
    print(HTML_REPORT)


def report_to_html(markdown_text: str) -> str:
    html_lines: list[str] = []
    in_table = False
    table_rows: list[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        header = table_rows[0]
        body = table_rows[2:] if len(table_rows) > 2 else []
        html_lines.append("<table>")
        html_lines.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr></thead>")
        html_lines.append("<tbody>")
        for row in body:
            html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        html_lines.append("</tbody></table>")
        in_table = False
        table_rows = []

    for line in markdown_text.splitlines():
        raw = line.strip()
        if raw.startswith("|") and raw.endswith("|"):
            cells = [c.strip() for c in raw.strip("|").split("|")]
            table_rows.append(cells)
            in_table = True
            continue
        flush_table()
        if raw.startswith("# "):
            html_lines.append(f"<h1>{raw[2:]}</h1>")
        elif raw.startswith("## "):
            html_lines.append(f"<h2>{raw[3:]}</h2>")
        elif raw.startswith("### "):
            html_lines.append(f"<h3>{raw[4:]}</h3>")
        elif raw.startswith("!["):
            alt = raw.split("](", 1)[0][2:]
            src = raw.split("](", 1)[1].rstrip(")")
            html_lines.append(f'<img src="{src}" alt="{alt}">')
        elif raw.startswith("*") and raw.endswith("*") and not raw.startswith("**"):
            html_lines.append(f"<em>{raw.strip('*')}</em>")
        elif raw.startswith("- "):
            html_lines.append(f"<p>• {raw[2:]}</p>")
        elif raw:
            html_lines.append(f"<p>{raw}</p>")
        else:
            html_lines.append("")
    flush_table()
    return "\n".join(html_lines)


if __name__ == "__main__":
    main()
