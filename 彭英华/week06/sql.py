import sqlite3

conn = sqlite3.connect("mydatabase.db")
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS students(
student_id INTEGER PRIMARY KEY,
name String NOT NULL,
class TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS classes(
class_id INTEGER PRIMARY KEY,
class_address TEXT NOT NULL
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS subjects(
subject_id INTEGER PRIMARY KEY,
subject_name TEXT NOT NULL,
student_id INTEGER,
FOREIGN KEY (student_id) REFERENCES students (student_id)
);
''')

conn.commit()
print("数据库和表已成功创建。")

# cursor.execute(
# "INSERT INTO students (student_id,name,class) VALUES(?,?,?)",(1,'李华','class_1'))
# cursor.execute("INSERT INTO students (student_id,name,class) VALUES (?,?,?)",(2,'小明','class_1'))
# conn.commit()
cursor.execute("SELECT student_id FROM students WHERE name ='李华'")
lihua_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO subjects (subject_name,student_id) VALUES (?,?)",('计算机',lihua_id))
cursor.execute("SELECT student_id FROM students WHERE name ='小明'")
xiaoming_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO subjects (subject_name,student_id) VALUES (?,?)",('高等数学',xiaoming_id))
print("\n--- 所有课程和学生 ---")
cursor.execute('''
SELECT subjects.subject_name, students.name
FROM subjects
JOIN students ON subjects.student_id = students.student_id;
''')
books_with_authors = cursor.fetchall()
for course, student in books_with_authors:
    print(f"学生: {student}, 课程: {course}")
print("\n--- 删除学生 ---")
cursor.execute("DELETE FROM students WHERE name = ?", ('xiaoming',))
conn.commit()
print("小明已被删除。")

print("\n--- 剩余学生 ---")
cursor.execute("SELECT name FROM students")
print(cursor.fetchone()[0]+"\n")
conn.close()
print("数据库关闭连接\n")
