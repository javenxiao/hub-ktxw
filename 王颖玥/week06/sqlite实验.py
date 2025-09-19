import sqlite3
import os

'''
针对员工信息数据分别用sqlite来实现数据的插入和查询
'''

# 每次运行前删除旧数据库
if os.path.exists('staff_changes.db'):
    os.remove('staff_changes.db')

conn = sqlite3.connect('staff_changes.db')
cursor = conn.cursor()

# 创建departments表
cursor.execute('''
CREATE TABLE IF NOT EXISTS departments(
    department_id INTEGER PRIMARY KEY,
    department_name TEXT NOT NULL UNIQUE,
    leader TEXT UNIQUE
    );
''')

# 创建employees表
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees(
    name TEXT NOT NULL, 
    gender TEXT NOT NULL CHECK (gender IN ('male', 'female')),
    department_id INTEGER,
    salary REAL CHECK (salary >= 0),
    entry_time TEXT,
    FOREIGN KEY (department_id) REFERENCES departments (department_id)
    );
''')

conn.commit()
print("数据库和表已成功创建。")

# 插入部门数据
cursor.execute("INSERT OR IGNORE INTO departments (department_name, leader) VALUES (?, ?)", ('Research and Development', 'Robert'))
cursor.execute("INSERT OR IGNORE INTO departments (department_name, leader) VALUES (?, ?)", ('Finance', 'Emma'))
cursor.execute("INSERT OR IGNORE INTO departments (department_name, leader) VALUES (?, ?)", ('Marketing', 'Olivia'))
conn.commit()

# 插入员工数据
cursor.execute("SELECT department_id FROM departments WHERE department_name = 'Research and Development'")
research_de_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO employees (name, gender, department_id, salary, entry_time) VALUES (?, ?, ?, ?, ?)", ('Tom', 'male', research_de_id, 20000.0, '2023-03-01'))

cursor.execute("SELECT department_id FROM departments WHERE department_name = 'Finance'")
finance_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO employees (name, gender, department_id, salary, entry_time) VALUES (?, ?, ?, ?, ?)", ('Jane', 'female', finance_id, 15000.0, '2024-12-18'))
cursor.execute("INSERT INTO employees (name, gender, department_id, salary, entry_time) VALUES (?, ?, ?, ?, ?)", ('John', 'male', finance_id, 8000.0, '2025-08-30'))

cursor.execute("SELECT department_id FROM departments WHERE department_name = 'Marketing'")
marketing_id = cursor.fetchone()[0]
cursor.execute("INSERT INTO employees (name, gender, department_id, salary, entry_time) VALUES (?, ?, ?, ?, ?)", ('Noah', 'male', marketing_id, 12000.0, '2025-05-02'))
conn.commit()

print("数据已成功插入。")

print("--- 所有部门及员工 ---")
cursor.execute('''
SELECT employees.name, employees.gender, departments.department_name
FROM employees
JOIN departments ON employees.department_id = departments.department_id;
''')

employees_with_departments = cursor.fetchall()
for name, gender, department in employees_with_departments:
    print(f"姓名：{name}, 性别：{gender}, 部门：{department}")

# 更新员工的工资
print("--- 更新员工工资 ---")
cursor.execute("UPDATE employees SET salary = ? WHERE name = ?", (18000.0, 'Jane'))
conn.commit()
print("员工'Jane'的工资已更新。")

# 查询更新后的数据
cursor.execute("SELECT name, salary FROM employees WHERE name = 'Jane'")
updated_employee = cursor.fetchone()
print(f"更新后的信息：姓名：{updated_employee[0]}, 工资：{updated_employee[1]}")

# 删除一个员工信息
print("--- 删除员工信息 ---")
cursor.execute("DELETE FROM employees WHERE name = ?", ('John',))
conn.commit()
print("员工'John'信息已被删除")

# 再次查询
print("--- 剩余的员工 ---")
# cursor.execute("SELECT name FROM employees")
cursor.execute('''
SELECT employees.name, employees.gender, departments.department_name
FROM employees
JOIN departments ON employees.department_id = departments.department_id;
''')
remaining_employee = cursor.fetchall()
for name, gender, department in remaining_employee:
    print(f"姓名：{name}, 性别：{gender}, 部门：{department}")

