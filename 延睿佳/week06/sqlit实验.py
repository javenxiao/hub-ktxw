import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('school.db')
cursor = conn.cursor()

# 创建学生表
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    major TEXT
);
''')

# 创建课程表
cursor.execute('''
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_name TEXT NOT NULL,
    teacher TEXT
);
''')

# 创建选课表（多对多关系）
cursor.execute('''
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    course_id INTEGER,
    grade REAL,
    FOREIGN KEY (student_id) REFERENCES students (student_id),
    FOREIGN KEY (course_id) REFERENCES courses (course_id)
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")

# 插入学生数据
students_data = [
    ("Alice", 20, "Computer Science"),
    ("Bob", 22, "Mathematics"),
    ("Charlie", 21, "Physics")
]
cursor.executemany("INSERT INTO students (name, age, major) VALUES (?, ?, ?)", students_data)
conn.commit()


# 插入课程数据
courses_data = [
    ("Database Systems", "Prof. Smith"),
    ("Linear Algebra", "Dr. Johnson"),
    ("Quantum Mechanics", "Dr. Brown")
]
cursor.executemany("INSERT INTO courses (course_name, teacher) VALUES (?, ?)", courses_data)
conn.commit()

# 插入选课数据
cursor.execute("INSERT INTO enrollments (student_id, course_id, grade) VALUES (1, 1, 88.5)")
cursor.execute("INSERT INTO enrollments (student_id, course_id, grade) VALUES (2, 2, 92.0)")
cursor.execute("INSERT INTO enrollments (student_id, course_id, grade) VALUES (3, 3, 85.0)")
conn.commit()
print("学生、课程和选课数据已插入。")

# 查询所有学生及其选课
print("\n--- 所有学生及其选课 ---")
cursor.execute('''
SELECT students.name, students.major, courses.course_name, enrollments.grade
FROM enrollments
JOIN students ON enrollments.student_id = students.student_id
JOIN courses ON enrollments.course_id = courses.course_id;
''')
for row in cursor.fetchall():
    print(f"学生: {row[0]}, 专业: {row[1]}, 课程: {row[2]}, 成绩: {row[3]}")

# 更新某个学生成绩
print("\n--- 更新成绩 ---")
cursor.execute("UPDATE enrollments SET grade = ? WHERE student_id = ? AND course_id = ?", (95.0, 1, 1))
conn.commit()
print("Alice 的 Database Systems 成绩已更新。")

# 删除一门课程
print("\n--- 删除课程 ---")
cursor.execute("DELETE FROM courses WHERE course_name = ?", ("Quantum Mechanics",))
conn.commit()
print("课程 'Quantum Mechanics' 已被删除。")

# 查询剩余课程
print("\n--- 剩余课程 ---")
cursor.execute("SELECT course_name, teacher FROM courses")
for row in cursor.fetchall():
    print(f"课程: {row[0]}, 授课教师: {row[1]}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
