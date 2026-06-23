import json
import threading
import time
from datetime import datetime

import redis
import kafka

'''
本地安装一下redis （或使用 http://onecompiler.com/redis/），学习redis的使用，学会操作list、set。
'''


def prac_redis():
    pool = redis.ConnectionPool(host='vm.orannet.icu', port=6379, password="orange", decode_responses=True, db=0)

    r = redis.Redis(connection_pool=pool)

    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"Redis连接失败:{e}")
        return

    with r.pipeline() as pipe:
        # 使用 SET 命令的 NX（不存在时才设置）和 EX（过期时间）参数
        pipe.set('ops_times', 0, ex=3600 * 24, nx=True)
        pipe.get('ops_times')
        result = pipe.execute()
        print("String操作结果:", result)

    def count_ops_times(func, *args, **kwargs):
        '''
        统计redis操作次数
        :param func:
        :param args:
        :param kwargs:
        :return:
        '''

        def wrapper(*args, **kwargs):
            r.incrby("ops_times", 1)
            return func(*args, **kwargs)

        return wrapper

    # list
    @count_ops_times
    def do_list():
        with r.pipeline() as pipe:
            pipe.delete("my_list")
            pipe.lpush("my_list", "item1")
            pipe.lpush("my_list", "item2")
            pipe.lpush("my_list", "item3")
            pipe.lrange("my_list", 0, -1)
            result = pipe.execute()
            print("List操作结果:", result)

    @count_ops_times
    def do_set():
        with r.pipeline() as pipe:
            pipe.delete("my_set*")
            pipe.sadd("my_set1", "item1")
            pipe.sadd("my_set2", "item1")
            pipe.sadd("my_set1", "item1")
            pipe.sadd("my_set1", "item3")
            pipe.sadd("my_set2", "item2")
            pipe.smembers("my_set1")
            pipe.smembers("my_set2")
            pipe.sunion("my_set1", "my_set2")
            pipe.sinter("my_set1", "my_set2")
            pipe.sdiff("my_set1", "my_set2")
            pipe.sdiff("my_set2", "my_set1")
            result = pipe.execute()
            print("my_set1 members:", result[-6])
            print("my_set2 members:", result[-5])
            print("Set操作结果, 以下结果分别是什么命令:", result[-4:])

    @count_ops_times
    def do_zset():
        with r.pipeline() as pipe:
            pipe.delete("my_zset*")
            pipe.zadd("my_zset1", {
                "python": 100,
                "java": 98,
                "c++": 95,
                "c": 99,
                "c#": 93,
                "javascript": 6
            })
            pipe.zrevrank("my_zset1", "java")
            pipe.zscore("my_zset1", "python")
            result = pipe.execute()
            print("Zset操作结果:", result)

    @count_ops_times
    def do_hash():
        with r.pipeline() as pipe:
            pipe.delete("my_hash*")
            mapping = {}
            for key, value in r.__dict__.items():
                if value is None:
                    continue  # 跳过None值

                # 处理布尔值：转换为字符串或整数
                if isinstance(value, bool):
                    mapping[key] = str(value).lower()  # 转换为 "true"/"false"
                    # 或者使用整数表示：mapping[key] = 1 if value else 0

                # 处理基本数据类型 str, int, float, bytes
                elif isinstance(value, (str, int, float, bytes)):
                    mapping[key] = value
            print(mapping)
            pipe.hset("my_hash1", mapping=mapping)
            pipe.hgetall("my_hash1")
            result = pipe.execute()
            print("Hash操作结果:", result)

    do_list()
    do_set()
    do_zset()
    do_hash()


def prac_kafka():
    TOPIC = "topic.orannet.icu"

    # 由于Python中print函数在多线程环境下不是原子操作导致的。
    # 虽然每个print调用会输出一个完整的字符串，但多个线程的print操作可能会相互穿插
    print_lock = threading.Lock()

    consumer = kafka.KafkaConsumer(
        TOPIC,
        bootstrap_servers=["vm.orannet.icu:9092"],
        auto_offset_reset='latest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    producer = kafka.KafkaProducer(
        bootstrap_servers=["vm.orannet.icu:9092"],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    def do_consume():
        print("启动消费者")
        for message in consumer:
            with print_lock:  # 获取锁
                print("消费者 ", datetime.now(), message)

    def do_produce(topic: str | list[str]):
        print("启动生产者")
        messages = "hello"
        future = producer.send(topic, {"message": messages})
        with print_lock:  # 获取锁
            print("生产者 ", future.get(timeout=10))

    t = threading.Thread(target=do_consume, daemon=True)
    t.start()

    # 等待消费者就绪
    time.sleep(3)

    do_produce(TOPIC)
    # t.join()

    time.sleep(5)


if __name__ == '__main__':
    prac_redis()
    # prac_kafka()
