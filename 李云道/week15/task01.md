## 作业一
> 阅读【08-financial-report-generator】中包含的方案，总结下其中使用了哪些外部数据源


### 方案-天池三轮车
- 巨潮抓取的A股财报,切分chunk后存储在阿里云的ES数据库
- 巨潮抓取的港股财报,切分chunk后存储在阿里云的ES数据库

### 方案-好想成为人类
- 来自AI回答的数据有：外部竞争对手，
- akshare 网上开放的股票数据接口：三大财务报表(资产负债表，利润表，现金流量表)，
- 行业信息的数据来源：k-sogou-search非官方搜狗引擎库，ddgs开源的元搜索库

### 方案-队伍名字不能为空
- 来自AI回答的数据有：公司信息，ROE、DCF、FCF分析
- 百度股市通API：三大财务报告
- 雪球API：三大财务报告
- 爬虫获取的国家统计局数据：经济数据
- 爬虫获取的中国人民银行：调查统计司数据
- akshare 网上开放的股票数据接口：外汇数据，期货数据，宏观经济数据，企业财务摘要，企业三大财务报告
- 爬虫获取的恒生指数
- 来自各大站点的新闻数据：华东师范Cube，中国新闻网
- 港交所报告：https://www.hkexnews.hk/index.htm
- 上交所报告：https://www.sse.com.cn/、
- 深交所报告：https://www.szse.cn/index/index.html
- 国务院政策文件：https://sousuo.www.gov.cn/
- 人民政府网：https://sousuoht.www.gov.cn
- 工信部网：https://www.miit.gov.cn/