import sqlite3

conn=sqlite3.connect('a_test.db')
cursor=conn.cursor()

cursor.execute('''
create table if not exists user(
    user_id INTEGER PRIMARY KEY,
    name VARCHAR(50) not null,
    age VARCHAR
);
''')

conn.commit()
print("数据库及用户表创建成功")

cursor.execute("delete from user")
conn.commit()
print("数据表已清空")

cursor.execute("Insert into user (name, age) values(?, ?)",("Kipper",10))
cursor.execute("Insert into user (name, age) values(?, ?)",("Alice",30))
cursor.execute("Insert into user (name, age) values(?, ?)",("Bob",20))
conn.commit()
print("数据已插入")

cursor.execute("select name,age from user")
users=cursor.fetchall()
for name,age in users:
    print(f"姓名：{name},年龄：{age}")

cursor.execute("update user set age=? where name=?",(35,"Alice"))
conn.commit()
cursor.execute("select name,age from user where name=?",("Alice",))
updated_name = cursor.fetchone()
print("数据已更新")
print(f"姓名：{updated_name[0]},更新后年龄：{updated_name[1]}")

cursor.execute("delete from user where name=?",("Bob",))
conn.commit()
print("Bob已删除")

cursor.execute("select name,age from user")
users=cursor.fetchall()
for name,age in users:
    print(f"姓名：{name},年龄：{age}")

conn.close()
print("数据库连接已关闭")