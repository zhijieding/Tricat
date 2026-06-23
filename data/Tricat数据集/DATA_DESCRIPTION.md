# Data Sources / 数据来源说明

## 1. Match Data & Betting Odds（比赛数据与博彩赔率数据）

**Source**: [Football-Data.co.uk](https://www.football-data.co.uk)

英格兰足球超级联赛（EPL）和法国足球甲级联赛（Ligue 1）的比赛数据（match data）及博彩公司赔率数据（betting odds data）均来源于公开数据平台 Football-Data。该网站提供可公开下载的历史足球比赛数据。

- **覆盖范围**：2015–2024年，共十个完整赛季
- **主要内容**：
  - 比赛结果（Full-Time Result）
  - 比赛技术统计（Match Statistics）
  - Bet365 等博彩公司赔率信息（Betting Odds）

## 2. Team Rating Data（球队能力评分数据）

**Source**: [SoFIFA](https://sofifa.com)

为进一步丰富特征空间并量化球队实力，本文引入同期 SoFIFA 提供的球队能力评分数据。SoFIFA 是基于 EA Sports FIFA / EA FC 系列游戏的公开球员与球队评分数据库，数据以网页形式免费公开。

### 数据获取方式

1. **网络爬虫采集**：采用网络爬虫对 SoFIFA 球队评分页面进行结构化爬取，每支球队每赛季对应唯一 URL（例如 `https://sofifa.com/teams?type=club&lg%5B0%5D=13&r=150059&set=true`）。
2. **覆盖范围**：2015年至2024年共10个赛季（对应 FIFA 15 至 FIFA 24 版本），每版游戏对应一个赛季的基础评分。
3. **时间对齐**：每赛季仅采用该赛季开始时的评分快照，确保特征反映赛季初的先验信息。
