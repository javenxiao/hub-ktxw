#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/9/18 07:47 
 Description:
作业1: sqlite、 orm、 es 的使用学习，自己构建数据插入
    实验sqlite 存储、检索不同的数据，截图
    实验es 存储、检索不同的数据，截图

作业2:RAG 实验代码，重新跑通。截图


"""

import sqlite3


# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('library.db')
cursor = conn.cursor()


# 创建 authors 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER,
    published_year INTEGER,
    FOREIGN KEY (author_id) REFERENCES authors (author_id)
);
''')

# 创建 borrowers 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")

# 插入作者数据
cursor.execute("INSERT INTO authors (name, nationality) VALUES (?, ?)", ('J.K. Rowling', 'British'))
cursor.execute("INSERT INTO authors (name, nationality) VALUES (?, ?)", ('George Orwell', 'British'))
cursor.execute("INSERT INTO authors (name, nationality) VALUES (?, ?)", ('Isaac Asimov', 'American'))
cursor.execute("insert into authors(name,nationality) values (?,?)",('zhangshan','china'))
conn.commit()

# 插入书籍数据
# J.K. Rowling 的 author_id 可能是 1，我们用 SELECT 查询来获取
cursor.execute("SELECT author_id FROM authors WHERE name = 'zhangshan'")
jk_rowling_id = cursor.fetchone()[0]
print(jk_rowling_id)
print('------')


cursor.execute("delete from books where title = ?",('Harry Potter',))
cursor.execute("delete from books where title = ?",('Ha',))
conn.commit()

cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)", ('Harry Potter', jk_rowling_id, 1997))
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)", ('Ha', jk_rowling_id, 1997))


cursor.execute("delete from books where title=?",('1984',))
conn.commit()
# 插入 George Orwell 的书籍
cursor.execute("SELECT author_id FROM authors WHERE name = 'George Orwell'")
george_orwell_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO books (title, author_id, published_year) VALUES (?, ?, ?)", ('1984', george_orwell_id, 1949))

conn.commit()



# 插入借阅人数据
# cursor.execute("INSERT INTO borrowers (name, email) VALUES (?, ?)", ('Alice', 'alice@example.com'))
# cursor.execute("INSERT INTO borrowers (name, email) VALUES (?, ?)", ('Bob', 'bob@example.com'))
# conn.commit()

print("数据已成功插入。")

# 查询所有书籍及其对应的作者名字
print("\n--- 所有书籍和它们的作者 ---")
cursor.execute('''
SELECT books.title, authors.name,books.published_year
FROM books
JOIN authors ON books.author_id = authors.author_id;
''')

book_with_authors = cursor.fetchall()
for book,author,published_year in book_with_authors:
    print(f'书籍：{book}，作者：{author},出版时间：{published_year}')


# 更新一本书的出版年份
print("\n--- 更新书籍信息 ---")
cursor.execute("UPDATE books SET published_year = ? WHERE title = ?", (1998, 'Harry Potter and the Sorcerer\'s Stone'))
conn.commit()
print("书籍 'Harry Potter and the Sorcerer\'s Stone' 的出版年份已更新。")

# 查询更新后的数据
cursor.execute("SELECT title, published_year FROM books WHERE title = 'Harry Potter'")
updated_book = cursor.fetchone()

print(f"更新后的信息: 书籍: {updated_book[0]}, 出版年份: {updated_book[1]}")

# 删除一个借阅人
print("\n--- 删除借阅人 ---")
cursor.execute("DELETE FROM borrowers WHERE name = ?", ('Bob',))
conn.commit()
print("借阅人 'Bob' 已被删除。")



# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余的借阅人 ---")
cursor.execute("SELECT name FROM borrowers")
remaining_borrowers = cursor.fetchall()
for borrower in remaining_borrowers:
    print(f"姓名: {borrower[0]}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
