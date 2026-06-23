import time

import redis


class RedisManager:

    def __init__(self):
        self.r = redis.Redis(
            host='localhost',
            port=6379,
            password="123456",  # 如果有密码
            db=0,  # 数据库编号
            decode_responses=True  # 自动解码为字符串
        )

    def check_link(self) -> bool:
        try:
            response = self.r.ping()
            print(f"Redis连接成功: {response}")
            return response
        except redis.exceptions.ConnectionError as e:
            print(f"连接失败: {e}")
            return False

    def test_single(self):
        if not self.check_link():
            return
        self.r.set('name', 'fengyihuan')
        self.r.set('age', 30)
        self.r.setex('token', 3, "xxxxx")
        print("info is -> ", self.r.get('name'), self.r.get('age'))
        print('token -> ', self.r.get('token'))
        time.sleep(4)
        print('token -> ', self.r.get('token'))

    def test_list(self):
        if not self.check_link():
            return
        list_key = 'numbers'
        self.r.delete(list_key)
        self.r.rpush(list_key, 1)
        self.r.rpush(list_key, 2)
        print('numbers -> ', self.r.lrange(list_key, 0, -1))
        self.r.lpush(list_key, 3)
        self.r.lpush(list_key, 4)
        print('numbers -> ', self.r.lrange(list_key, 0, -1))
        print('lpop -> ', self.r.lpop(list_key))
        print('numbers -> ', self.r.lrange(list_key, 0, -1))
        print('rpop -> ', self.r.rpop(list_key))
        print('numbers -> ', self.r.lrange(list_key, 0, -1))
        print("lenght -> ", self.r.llen(list_key))

    def test_hash(self):
        if not self.check_link():
            return
        hash_key = "user:1000"
        self.r.delete(hash_key)
        self.r.hset(hash_key, "name", "fengyihuan")
        self.r.hset(hash_key, "age", "30")
        self.r.hmset(hash_key, mapping={"sex": "man", "email": "xx@xx.com"})
        print('age ->', self.r.hget(hash_key, "age"))
        print('user 1000 -> ', self.r.hgetall(hash_key))
        self.r.hset(hash_key, "age", "32")
        print('user 1000 -> ', self.r.hgetall(hash_key))
        print("check email exist ->", self.r.hexists(hash_key, "email"))

    def test_set(self):
        if not self.check_link():
            return
        set_key_1 = "set_key_1"
        set_key_2 = "set_key_2"
        self.r.sadd(set_key_1, 'python', 'java', 'kotlin')
        self.r.sadd(set_key_1, 'c', 'java', 'swift')
        print('set -> ', self.r.smembers(set_key_1))
        self.r.sadd(set_key_2, 'object-c', 'java', 'javascript')
        print('set -> ', self.r.smembers(set_key_2))
        inter = self.r.sinter(set_key_1, set_key_2)
        union = self.r.sunion(set_key_1, set_key_2)
        diff = self.r.sdiff(set_key_1, set_key_2)
        print(f"inter -> {inter}")
        print(f"union -> {union}")
        print(f"diff -> {diff}")


if __name__ == '__main__':
    manager = RedisManager()
    # manager.test_single()
    # manager.test_list()
    # manager.test_hash()
    manager.test_set()


