
# 数据
DATAS = [
    {
        "doc_id": "DOC2024090001",
        "title": "《ElasticSearch 8.x 全文检索实战指南》",
        "content": "本书详细讲解 ElasticSearch 8.x 版本的核心功能，包括索引设计、分词器配置、复杂查询编写和集群优化。适合零基础开发者快速入门，也可作为中级工程师的进阶参考。书中包含 50+ 实战案例，覆盖电商商品搜索、日志分析等场景。",
        "category": "技术文档",
        "author": "张三",
        "publish_date": "2024-01-15",
        "views": 3280,
        "tags": ["ElasticSearch", "全文检索", "实战教程"],
        "is_recommended": True
    },
    {
        "doc_id": "DOC2024090002",
        "title": "《Python 数据分析从入门到精通》",
        "content": "以 Pandas、NumPy、Matplotlib 为核心工具，从数据清洗、特征工程到可视化分析，逐步讲解数据分析全流程。书中案例基于真实业务数据（电商用户行为、金融交易记录），配套代码可直接运行，帮助读者快速掌握实战技能。",
        "category": "技术文档",
        "author": "李四",
        "publish_date": "2023-11-08",
        "views": 5120,
        "tags": ["Python", "数据分析", "Pandas"],
        "is_recommended": True
    },
    {
        "doc_id": "DOC2024090003",
        "title": "《2024 电商平台运营策略白皮书》",
        "content": "基于 2023 年电商行业数据，分析直播带货、私域流量、会员体系三大核心运营模式的效果。提出“精准用户分层+场景化营销”的新策略，配套 10 个头部电商案例（如淘宝、京东）的实操方案，适合运营管理者参考。",
        "category": "运营文档",
        "author": "王五",
        "publish_date": "2024-03-20",
        "views": 2850,
        "tags": ["电商运营", "营销策略", "行业报告"],
        "is_recommended": False
    },
    {
        "doc_id": "DOC2024090004",
        "title": "《Java SpringBoot 微服务架构设计手册》",
        "content": "从单体应用拆分到微服务部署，讲解 SpringBoot 核心注解、SpringCloud 组件（Eureka、Gateway、Feign）的使用。重点覆盖服务注册发现、配置中心、熔断降级等关键技术，附带完整的微服务项目源码（GitHub 可下载）。",
        "category": "技术文档",
        "author": "赵六",
        "publish_date": "2023-09-30",
        "views": 4680,
        "tags": ["Java", "SpringBoot", "微服务"],
        "is_recommended": True
    },
    {
        "doc_id": "DOC2024090005",
        "title": "《产品需求文档（PRD）撰写规范》",
        "content": "明确 PRD 的核心结构（需求背景、功能描述、交互原型、验收标准），提供 5 类常见产品（APP、小程序、后台系统）的 PRD 模板。讲解如何用 Axure 绘制交互原型，以及如何与研发、设计团队对齐需求，避免沟通偏差。",
        "category": "产品文档",
        "author": "孙七",
        "publish_date": "2024-02-12",
        "views": 1980,
        "tags": ["产品设计", "PRD", "需求管理"],
        "is_recommended": False
    },
    # {
    #     "doc_id": "DOC2024090006",
    #     "title": "《MySQL 8.0 索引优化与性能调优》",
    #     "content": "深入讲解 MySQL 索引原理（B+树、哈希索引），分析慢查询日志的优化方法。包含 15 个常见性能问题案例（如索引失效、锁等待），提供 SQL 语句优化技巧和服务器参数配置建议，帮助读者将数据库 QPS 提升 3-5 倍。",
    #     "category": "技术文档",
    #     "author": "张三",
    #     "publish_date": "2023-12-05",
    #     "views": 3920,
    #     "tags": ["MySQL", "索引优化", "性能调优"],
    #     "is_recommended": True
    # },
    # {
    #     "doc_id": "DOC2024090007",
    #     "title": "《2024 新媒体营销渠道效果对比报告》",
    #     "content": "对比抖音、小红书、视频号、B站四大新媒体渠道的用户画像、流量成本和转化效率。基于 2024 年第一季度数据，给出不同行业（美妆、3C、快消）的渠道选择建议，附带 3 个成功案例的投放策略拆解。",
    #     "category": "市场文档",
    #     "author": "周八",
    #     "publish_date": "2024-04-18",
    #     "views": 1750,
    #     "tags": ["新媒体", "营销渠道", "效果分析"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090008",
    #     "title": "《Vue3 + Vite 前端项目实战》",
    #     "content": "以一个电商前台项目为例，讲解 Vue3 组合式 API、Pinia 状态管理、Vue Router 路由配置的使用。介绍 Vite 的构建优化技巧和 Element Plus 组件库的实战应用，最后讲解项目部署到 Nginx 和 Docker 的完整流程。",
    #     "category": "技术文档",
    #     "author": "吴九",
    #     "publish_date": "2024-01-30",
    #     "views": 2680,
    #     "tags": ["Vue3", "Vite", "前端开发"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090009",
    #     "title": "《员工招聘流程管理规范（2024 版）》",
    #     "content": "明确招聘全流程（需求审批、简历筛选、面试安排、Offer 发放）的责任部门和时间节点。提供不同岗位（技术、产品、运营）的面试评分表和简历模板，强调背景调查和入职引导的关键步骤，确保招聘效率和人才质量。",
    #     "category": "人事文档",
    #     "author": "郑十",
    #     "publish_date": "2024-03-05",
    #     "views": 980,
    #     "tags": ["人事制度", "招聘流程", "管理规范"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090010",
    #     "title": "《ElasticSearch 向量搜索实战：AI 文档问答系统》",
    #     "content": "结合 OpenAI Embedding 模型，讲解如何将文档转换为向量并存储到 ElasticSearch。实现“用户提问→向量生成→相似文档匹配→答案生成”的完整流程，配套 Python 代码（使用 elasticsearch-py 和 openai 库），适合 AI+搜索场景落地。",
    #     "category": "技术文档",
    #     "author": "李四",
    #     "publish_date": "2024-05-10",
    #     "views": 1850,
    #     "tags": ["ElasticSearch", "向量搜索", "AI 问答"],
    #     "is_recommended": True
    # },
    # {
    #     "doc_id": "DOC2024090011",
    #     "title": "《电商商品详情页 UI 设计规范》",
    #     "content": "从用户体验角度，规范商品详情页的布局（主图区、参数区、评价区、推荐区）、字体大小和颜色搭配。提供移动端和 PC 端的设计模板，强调加载速度优化（如图片懒加载）和转化率提升技巧（如突出优惠信息）。",
    #     "category": "产品文档",
    #     "author": "孙七",
    #     "publish_date": "2023-10-22",
    #     "views": 1260,
    #     "tags": ["UI 设计", "电商产品", "用户体验"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090012",
    #     "title": "《Python 爬虫实战：爬取豆瓣电影Top250数据》",
    #     "content": "详细讲解 Requests 库发送请求、BeautifulSoup 解析 HTML、Selenium 处理动态页面的方法。实现自动爬取豆瓣电影的名称、评分、导演、剧情简介等数据，存储到 CSV 文件并进行简单分析（如评分分布统计），附带反爬策略（如设置请求头、IP 代理）。",
    #     "category": "技术文档",
    #     "author": "吴九",
    #     "publish_date": "2023-08-15",
    #     "views": 4210,
    #     "tags": ["Python", "爬虫", "数据采集"],
    #     "is_recommended": True
    # },
    # {
    #     "doc_id": "DOC2024090013",
    #     "title": "《2024 年第一季度运营工作总结》",
    #     "content": "总结 Q1 运营工作成果（用户增长 20%、GMV 提升 15%），分析直播带货和优惠券活动的效果。指出存在的问题（如复购率偏低），提出 Q2 改进计划（如会员专属活动、社群运营），附带关键数据图表（用户增长曲线、渠道转化漏斗）。",
    #     "category": "运营文档",
    #     "author": "王五",
    #     "publish_date": "2024-04-02",
    #     "views": 850,
    #     "tags": ["运营总结", "季度报告", "数据复盘"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090014",
    #     "title": "《Java 并发编程：线程池原理与实战》",
    #     "content": "深入讲解 Java 线程池的核心参数（核心线程数、最大线程数、队列容量）和工作原理。分析常见问题（如线程泄漏、死锁），提供线程池优化建议（如根据业务场景选择拒绝策略），附带线程池监控和调优的实战代码。",
    #     "category": "技术文档",
    #     "author": "赵六",
    #     "publish_date": "2023-11-18",
    #     "views": 3120,
    #     "tags": ["Java", "并发编程", "线程池"],
    #     "is_recommended": True
    # },
    # {
    #     "doc_id": "DOC2024090015",
    #     "title": "《品牌社交媒体内容营销方案》",
    #     "content": "针对年轻用户群体，设计社交媒体内容矩阵（抖音短视频、小红书图文、B站长视频）。规划内容主题（产品测评、用户故事、行业知识）和发布节奏，制定 KPI 考核标准（播放量、互动率、转化率），预算分配和效果追踪方法。",
    #     "category": "市场文档",
    #     "author": "周八",
    #     "publish_date": "2024-02-25",
    #     "views": 1480,
    #     "tags": ["品牌营销", "内容运营", "社交媒体"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090016",
    #     "title": "《ElasticSearch 集群部署与监控指南》",
    #     "content": "讲解 ElasticSearch 集群的搭建步骤（3 节点配置、分片和副本分配），使用 Kibana 进行集群监控（查看节点状态、索引健康度）。介绍常见故障处理（节点下线、分片未分配），提供集群性能优化建议（如 JVM 内存配置、磁盘空间管理）。",
    #     "category": "技术文档",
    #     "author": "张三",
    #     "publish_date": "2024-03-18",
    #     "views": 2750,
    #     "tags": ["ElasticSearch", "集群部署", "监控"],
    #     "is_recommended": True
    # },
    # {
    #     "doc_id": "DOC2024090017",
    #     "title": "《产品迭代规划：V2.0 版本需求清单》",
    #     "content": "明确 V2.0 版本的核心目标（提升用户留存率），列出功能需求（如个性化推荐、消息推送、积分体系）。标注需求优先级（P0/P1/P2）和开发周期，提供需求对应的用户故事和验收标准，便于研发团队排期。",
    #     "category": "产品文档",
    #     "author": "孙七",
    #     "publish_date": "2024-01-08",
    #     "views": 1120,
    #     "tags": ["产品迭代", "需求清单", "版本规划"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090018",
    #     "title": "《员工绩效考核制度（2024 修订版）》",
    #     "content": "调整绩效考核指标（KPI+OKR 结合），明确不同岗位（技术、产品、运营）的考核权重。规范考核流程（自评、上级评、跨部门评）和周期（季度+年度），说明考核结果与薪酬、晋升的关联规则，确保公平性和激励性。",
    #     "category": "人事文档",
    #     "author": "郑十",
    #     "publish_date": "2024-02-15",
    #     "views": 780,
    #     "tags": ["人事制度", "绩效考核", "薪酬体系"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090019",
    #     "title": "《Python 自动化测试：Selenium + Pytest 实战》",
    #     "content": "讲解如何用 Selenium 模拟浏览器操作，编写自动化测试用例（登录、下单、支付流程）。使用 Pytest 进行用例管理和断言，生成测试报告（HTML 格式），实现持续集成（Jenkins 触发自动化测试），提高测试效率。",
    #     "category": "技术文档",
    #     "author": "吴九",
    #     "publish_date": "2023-09-10",
    #     "views": 2350,
    #     "tags": ["Python", "自动化测试", "Selenium"],
    #     "is_recommended": False
    # },
    # {
    #     "doc_id": "DOC2024090020",
    #     "title": "《电商大促活动（618）运营执行方案》",
    #     "content": "规划 618 大促的全周期（蓄水期、预热期、爆发期、返场期）活动策略，明确各阶段的营销重点（如蓄水期拉新、爆发期转化）。提供活动页面设计要求、客服排班表、物流应急预案，附带预算分配和效果预估（GMV 目标 5000 万）。",
    #     "category": "运营文档",
    #     "author": "王五",
    #     "publish_date": "2024-05-20",
    #     "views": 1680,
    #     "tags": ["电商大促", "618活动", "执行方案"],
    #     "is_recommended": True
    # }
]