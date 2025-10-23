import logging # 日志打印模块
#https://fcnbisyf4ls8.feishu.cn/wiki/ZNO2wypXfiulT0ks9cicuw6RnLr?from=from_copylink
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # 输出到文件
        logging.StreamHandler(),         # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)
#https://fcnbisyf4ls8.feishu.cn/wiki/Oya6wukEeideqtkbKFCcOOw5nBh?from=from_copylink
#https://fcnbisyf4ls8.feishu.cn/wiki/UErEw01gMiOttVka2UmcX9fFnnh?from=from_copylink