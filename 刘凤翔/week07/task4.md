## 🐳 Docker 简介

### 什么是 Docker？
Docker 是一个开源的应用容器引擎，让你可以将应用程序及其依赖打包成轻量级、可移植的容器。

### 核心概念：
- **镜像(Image)**：应用的模板，像虚拟机镜像
- **容器(Container)**：镜像的运行实例，像轻量级虚拟机
- **仓库(Registry)**：存储镜像的地方，如 Docker Hub

## 💻 Docker 安装指南
### macOS 系统安装
```bash
# 1. 下载 Docker Desktop for Mac
# 访问：https://docs.docker.com/desktop/install/mac-install/

# 2. 安装后验证
docker --version
docker run hello-world
```

### Linux 系统安装（Ubuntu为例）
```bash
# 1. 卸载旧版本
sudo apt remove docker docker-engine docker.io containerd runc

# 2. 安装依赖
sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release

# 3. 添加 Docker 官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 4. 添加仓库
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 安装 Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# 6. 验证安装
sudo docker run hello-world

# 7. 将用户添加到 docker 组（避免每次使用 sudo）
sudo usermod -aG docker $USER
# 重新登录生效
```

## 🔧 常用 Docker 命令

### 基础命令
```bash
# 查看版本
docker --version

# 查看系统信息
docker info

# 拉取镜像
docker pull nginx:latest

# 运行容器
docker run -d -p 80:80 --name my-nginx nginx

# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 停止容器
docker stop my-nginx

# 启动容器
docker start my-nginx

# 进入容器
docker exec -it my-nginx bash

# 查看日志
docker logs my-nginx

# 删除容器
docker rm my-nginx

# 删除镜像
docker rmi nginx
```

### 镜像管理
```bash
# 查看本地镜像
docker images

# 构建镜像
docker build -t my-app .

# 推送镜像到仓库
docker push username/my-app:latest
```

## 📁 Dockerfile 示例

创建一个简单的 Dockerfile：
```dockerfile
# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器中的 /app
COPY . /app

# 安装依赖
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 5000

# 定义环境变量
ENV NAME World

# 容器启动时运行
CMD ["python", "app.py"]
```

## 🚀 为 Dify 部署做准备

### 1. 创建专用目录
```bash
mkdir dify-docker && cd dify-docker
```

### 2. 准备 docker-compose.yml
Dify 通常使用 Docker Compose 部署，创建配置文件：
```yaml
# docker-compose.yml
version: '3.8'

services:
  dify:
    image: langgenius/dify-community:latest
    ports:
      - "5001:5001"
    environment:
      - DB_TYPE=sqlite
    volumes:
      - ./data:/app/data
```

### 3. 测试 Docker 安装
```bash
# 测试 Docker 运行正常
docker run -d -p 8080:80 --name test-nginx nginx
# 访问 http://localhost:8080 应该看到 Nginx 欢迎页面

# 停止测试容器
docker stop test-nginx
docker rm test-nginx
```

## 🛠️ 故障排查

### 常见问题解决

**权限问题（Linux）：**
```bash
sudo usermod -aG docker $USER
# 重新登录后生效
```

**端口冲突：**
```bash
# 查看端口占用
netstat -tulpn | grep :5001
# 更改 docker-compose.yml 中的端口映射
```

**磁盘空间不足：**
```bash
# 清理无用镜像和容器
docker system prune
```
