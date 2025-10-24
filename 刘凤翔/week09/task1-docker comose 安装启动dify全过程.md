# 一、docker安装dify全过程

Last login: Fri Oct 10 15:56:36 on ttys003

(base) liufengxiang@liufengangdeMBP ~ % git clone https://github.com/langgenius/dify.git --branch 0.15.3       

正克隆到 'dify'...

错误：RPC 失败。curl 55 Failed sending data to the peer

致命错误：在引用列表之后应该有一个 flush 包

(base) liufengxiang@liufengangdeMBP ~ % git clone https://github.com/langgenius/dify.git --branch 0.15.3

正克隆到 'dify'...

错误：RPC 失败。curl 55 Failed sending data to the peer

致命错误：在引用列表之后应该有一个 flush 包

(base) liufengxiang@liufengangdeMBP ~ % ping github.com

PING github.com (20.205.243.166): 56 data bytes

64 bytes from 20.205.243.166: icmp_seq=0 ttl=112 time=98.508 ms

64 bytes from 20.205.243.166: icmp_seq=1 ttl=112 time=97.800 ms

64 bytes from 20.205.243.166: icmp_seq=2 ttl=112 time=98.746 ms

64 bytes from 20.205.243.166: icmp_seq=3 ttl=112 time=102.283 ms

64 bytes from 20.205.243.166: icmp_seq=4 ttl=112 time=101.724 ms

64 bytes from 20.205.243.166: icmp_seq=5 ttl=112 time=99.394 ms

64 bytes from 20.205.243.166: icmp_seq=6 ttl=112 time=101.964 ms

64 bytes from 20.205.243.166: icmp_seq=7 ttl=112 time=97.833 ms

64 bytes from 20.205.243.166: icmp_seq=8 ttl=112 time=97.830 ms

^C

--- github.com ping statistics ---

10 packets transmitted, 9 packets received, 10.0% packet loss

round-trip min/avg/max/stddev = 97.800/99.565/102.283/1.787 ms

(base) liufengxiang@liufengangdeMBP ~ % curl -I https://github.com/langgenius/dify









^C

(base) liufengxiang@liufengangdeMBP ~ % git config --list | grep -E "(proxy|buffer|ssl)"

(base) liufengxiang@liufengangdeMBP ~ % ls

Applications			Public

CodeGeeXProjects		miniconda3

Desktop				models

Documents			player.list

Downloads			requirements.txt

Library				stable-diffusion-webui

Movies				stable-diffusion-webui_坏的

Music				venv

Pictures

(base) liufengxiang@liufengangdeMBP ~ % git clone --depth 1 https://github.com/langgenius/dify.git --branch 0.15.3

正克隆到 'dify'...

错误：RPC 失败。curl 55 Failed sending data to the peer

致命错误：在引用列表之后应该有一个 flush 包

(base) liufengxiang@liufengangdeMBP ~ % ls                  

Applications			Public

CodeGeeXProjects		miniconda3

Desktop				models

Documents			player.list

Downloads			requirements.txt

Library				stable-diffusion-webui

Movies				stable-diffusion-webui_坏的

Music				venv

Pictures

(base) liufengxiang@liufengangdeMBP ~ % ls

Applications			Public

CodeGeeXProjects		dify_test

Desktop				miniconda3

Documents			models

Downloads			player.list

Library				requirements.txt

Movies				stable-diffusion-webui

Music				stable-diffusion-webui_坏的

Pictures			venv

(base) liufengxiang@liufengangdeMBP ~ % cd dify_test

(base) liufengxiang@liufengangdeMBP dify_test % curl -L -o dify-0.15.3.zip https://github.com/langgenius/dify/archive/refs/heads/main.zip

 % Total  % Received % Xferd Average Speed  Time  Time   Time Current

​                 Dload Upload  Total  Spent  Left Speed

 0   0  0   0  0   0   0   0 --:--:-- 0:00:19 --:--:--   0

100 23.6M  0 23.6M  0   0 1016k   0 --:--:-- 0:00:23 --:--:-- 7011k

(base) liufengxiang@liufengangdeMBP dify_test % ls

dify-0.15.3.zip

(base) liufengxiang@liufengangdeMBP dify_test % cd dify-0.15.3

cd: no such file or directory: dify-0.15.3

(base) liufengxiang@liufengangdeMBP dify_test % ls

dify-0.15.3.zip	dify-main

(base) liufengxiang@liufengangdeMBP dify_test % cd dify-main

(base) liufengxiang@liufengangdeMBP dify-main % ls

AGENTS.md	CONTRIBUTING.md	README.md	docker		scripts

AUTHORS		LICENSE		api		docs		sdks

CLAUDE.md	Makefile	dev		images		web

(base) liufengxiang@liufengangdeMBP dify-main % cd docker

(base) liufengxiang@liufengangdeMBP docker % ls

README.md			generate_docker_compose

certbot				middleware.env.example

couchbase-server		nginx

docker-compose-template.yaml	pgvector

docker-compose.middleware.yaml	ssrf_proxy

docker-compose.png		startupscripts

docker-compose.yaml		tidb

elasticsearch			volumes

(base) liufengxiang@liufengangdeMBP docker % cp .env.example .env

(base) liufengxiang@liufengangdeMBP docker % ls

README.md			generate_docker_compose

certbot				middleware.env.example

couchbase-server		nginx

docker-compose-template.yaml	pgvector

docker-compose.middleware.yaml	ssrf_proxy

docker-compose.png		startupscripts

docker-compose.yaml		tidb

elasticsearch			volumes

(base) liufengxiang@liufengangdeMBP docker % $ docker compose version

zsh: command not found: $

(base) liufengxiang@liufengangdeMBP docker % docker compose version 

docker: unknown command: docker compose



Run 'docker --help' for more information

(base) liufengxiang@liufengangdeMBP docker % docker --help

Usage: docker [OPTIONS] COMMAND



A self-sufficient runtime for containers



Common Commands:

 run     Create and run a new container from an image

 exec    Execute a command in a running container

 ps     List containers

 build    Build an image from a Dockerfile

 pull    Download an image from a registry

 push    Upload an image to a registry

 images   List images

 login    Authenticate to a registry

 logout   Log out from a registry

 search   Search Docker Hub for images

 version   Show the Docker version information

 info    Display system-wide information



Management Commands:

 builder   Manage builds

 checkpoint Manage checkpoints

 container  Manage containers

 context   Manage contexts

 image    Manage images

 manifest  Manage Docker image manifests and manifest lists

 network   Manage networks

 plugin   Manage plugins

 system   Manage Docker

 trust    Manage trust on Docker images

 volume   Manage volumes



Swarm Commands:

 config   Manage Swarm configs

 node    Manage Swarm nodes

 secret   Manage Swarm secrets

 service   Manage Swarm services

 stack    Manage Swarm stacks

 swarm    Manage Swarm



Commands:

 attach   Attach local standard input, output, and error streams to a running container

 commit   Create a new image from a container's changes

 cp     Copy files/folders between a container and the local filesystem

 create   Create a new container

 diff    Inspect changes to files or directories on a container's filesystem

 events   Get real time events from the server

 export   Export a container's filesystem as a tar archive

 history   Show the history of an image

 import   Import the contents from a tarball to create a filesystem image

 inspect   Return low-level information on Docker objects

 kill    Kill one or more running containers

 load    Load an image from a tar archive or STDIN

 logs    Fetch the logs of a container

 pause    Pause all processes within one or more containers

 port    List port mappings or a specific mapping for the container

 rename   Rename a container

 restart   Restart one or more containers

 rm     Remove one or more containers

 rmi     Remove one or more images

 save    Save one or more images to a tar archive (streamed to STDOUT by default)

 start    Start one or more stopped containers

 stats    Display a live stream of container(s) resource usage statistics

 stop    Stop one or more running containers

 tag     Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE

 top     Display the running processes of a container

 unpause   Unpause all processes within one or more containers

 update   Update configuration of one or more containers

 wait    Block until one or more containers stop, then print their exit codes



Global Options:

   --config string   Location of client config files (default

​              "/Users/liufengxiang/.docker")

 -c, --context string   Name of the context to use to connect to the

​              daemon (overrides DOCKER_HOST env var and

​              default context set with "docker context use")

 -D, --debug       Enable debug mode

 -H, --host string    Daemon socket to connect to

 -l, --log-level string  Set the logging level ("debug", "info",

​              "warn", "error", "fatal") (default "info")

   --tls        Use TLS; implied by --tlsverify

   --tlscacert string  Trust certs signed only by this CA (default

​              "/Users/liufengxiang/.docker/ca.pem")

   --tlscert string   Path to TLS certificate file (default

​              "/Users/liufengxiang/.docker/cert.pem")

   --tlskey string   Path to TLS key file (default

​              "/Users/liufengxiang/.docker/key.pem")

   --tlsverify     Use TLS and verify the remote

 -v, --version      Print version information and quit



Run 'docker COMMAND --help' for more information on a command.



For more help on how to use Docker, head to https://docs.docker.com/go/guides/

(base) liufengxiang@liufengangdeMBP docker % docker-compose version

zsh: command not found: docker-compose

(base) liufengxiang@liufengangdeMBP docker % docker compose version

docker: unknown command: docker compose



Run 'docker --help' for more information

(base) liufengxiang@liufengangdeMBP docker % docker --version

Docker version 28.4.0, build d8eb465

(base) liufengxiang@liufengangdeMBP docker % docker info

Client:

 Version:  28.4.0

 Context:  default

 Debug Mode: false



Server:

Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?

(base) liufengxiang@liufengangdeMBP docker % which docker-compose

docker-compose not found

(base) liufengxiang@liufengangdeMBP docker % docker info

Client:

 Version:  28.4.0

 Context:  desktop-linux

 Debug Mode: false

 Plugins:

 ai: Docker AI Agent - Ask Gordon (Docker Inc.)

  Version: v1.9.11

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-ai

 buildx: Docker Buildx (Docker Inc.)

  Version: v0.28.0-desktop.1

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-buildx

 cloud: Docker Cloud (Docker Inc.)

  Version: v0.4.27

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-cloud

 compose: Docker Compose (Docker Inc.)

  Version: v2.39.2-desktop.1

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-compose

 debug: Get a shell into any image or container (Docker Inc.)

  Version: 0.0.42

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-debug

 desktop: Docker Desktop commands (Docker Inc.)

  Version: v0.2.0

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-desktop

 extension: Manages Docker extensions (Docker Inc.)

  Version: v0.2.31

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-extension

 init: Creates Docker-related starter files for your project (Docker Inc.)

  Version: v1.4.0

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-init

 mcp: Docker MCP Plugin (Docker Inc.)

  Version: v0.18.0

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-mcp

 model: Docker Model Runner (Docker Inc.)

  Version: v0.1.40

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-model

 sbom: View the packaged-based Software Bill Of Materials (SBOM) for an image (Anchore Inc.)

  Version: 0.6.0

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-sbom

 scout: Docker Scout (Docker Inc.)

  Version: v1.18.3

  Path:   /Users/liufengxiang/.docker/cli-plugins/docker-scout



Server:

 Containers: 0

 Running: 0

 Paused: 0

 Stopped: 0

 Images: 0

 Server Version: 28.4.0

 Storage Driver: overlayfs

 driver-type: io.containerd.snapshotter.v1

 Logging Driver: json-file

 Cgroup Driver: cgroupfs

 Cgroup Version: 2

 Plugins:

 Volume: local

 Network: bridge host ipvlan macvlan null overlay

 Log: awslogs fluentd gcplogs gelf journald json-file local splunk syslog

 CDI spec directories:

 /etc/cdi

 /var/run/cdi

 Discovered Devices:

 cdi: docker.com/gpu=webgpu

 Swarm: inactive

 Runtimes: runc io.containerd.runc.v2

 Default Runtime: runc

 Init Binary: docker-init

 containerd version: 05044ec0a9a75232cad458027ca83437aae3f4da

 runc version: v1.2.5-0-g59923ef

 init version: de40ad0

 Security Options:

 seccomp

  Profile: builtin

 cgroupns

 Kernel Version: 6.10.14-linuxkit

 Operating System: Docker Desktop

 OSType: linux

 Architecture: aarch64

 CPUs: 14

 Total Memory: 7.653GiB

 Name: docker-desktop

 ID: 08601ae6-f272-4388-b221-ec74839917ee

 Docker Root Dir: /var/lib/docker

 Debug Mode: false

 HTTP Proxy: http.docker.internal:3128

 HTTPS Proxy: http.docker.internal:3128

 No Proxy: hubproxy.docker.internal

 Labels:

 com.docker.desktop.address=unix:///Users/liufengxiang/Library/Containers/com.docker.docker/Data/docker-cli.sock

 Experimental: false

 Insecure Registries:

 hubproxy.docker.internal:5555

 ::1/128

 127.0.0.0/8

 Live Restore Enabled: false



(base) liufengxiang@liufengangdeMBP docker % docker compose version

Docker Compose version v2.39.2-desktop.1

(base) liufengxiang@liufengangdeMBP docker % docker compose up -d

[+] Running 84/86

 ✔ api Pulled                              215.1s 

 ✔ web Pulled                              134.5s 

 ✔ weaviate Pulled                            102.3s 

 ✔ db Pulled                               167.2s 

 ✔ sandbox Pulled                            137.0s 

 ✔ worker_beat Pulled                          215.1s 

 ✔ ssrf_proxy Pulled                           99.6s 

 ✔ plugin_daemon Pulled                         192.0s 

 ✔ nginx Pulled                             121.9s 

 ✔ worker Pulled                             215.2s 

 ✔ redis Pulled                             105.7s 

[+] Running 13/13

 ✔ Network docker_ssrf_proxy_network Created               0.0s 

 ✔ Network docker_default       Created               0.0s 

 ✔ Container docker-weaviate-1    Sta...                1.0s 

 ✔ Container docker-redis-1      Starte...              1.0s 

 ✔ Container docker-ssrf_proxy-1   S...                 1.0s 

 ✔ Container docker-web-1       Started               1.0s 

 ✔ Container docker-sandbox-1     Star...               1.0s 

 ✔ Container docker-db-1       Healthy               4.5s 

 ✔ Container docker-worker_beat-1   Started               4.0s 

 ✔ Container docker-worker-1     Start...               4.0s 

 ✔ Container docker-plugin_daemon-1  Started               4.0s 

 ✔ Container docker-api-1       Started               4.1s 

 ✔ Container docker-nginx-1      Starte...              4.1s 

(base) liufengxiang@liufengangdeMBP docker % docker compose ps

NAME           IMAGE                    COMMAND          SERVICE     CREATED     STATUS          PORTS

docker-api-1       langgenius/dify-api:1.9.2          "/bin/bash /entrypoi…"  api       34 seconds ago  Up 30 seconds       5001/tcp

docker-db-1       postgres:15-alpine             "docker-entrypoint.s…"  db       34 seconds ago  Up 33 seconds (healthy)  5432/tcp

docker-nginx-1      nginx:latest                "sh -c 'cp /docker-e…"  nginx      34 seconds ago  Up 30 seconds       0.0.0.0:80->80/tcp, [::]:80->80/tcp, 0.0.0.0:443->443/tcp, [::]:443->443/tcp

docker-plugin_daemon-1  langgenius/dify-plugin-daemon:0.3.3-local  "/bin/bash -c /app/e…"  plugin_daemon  34 seconds ago  Up 30 seconds       0.0.0.0:5003->5003/tcp, [::]:5003->5003/tcp

docker-redis-1      redis:6-alpine               "docker-entrypoint.s…"  redis      34 seconds ago  Up 33 seconds (healthy)  6379/tcp

docker-sandbox-1     langgenius/dify-sandbox:0.2.12       "/main"          sandbox     34 seconds ago  Up 33 seconds (healthy)  

docker-ssrf_proxy-1   ubuntu/squid:latest             "sh -c 'cp /docker-e…"  ssrf_proxy   34 seconds ago  Up 33 seconds       3128/tcp

docker-weaviate-1    semitechnologies/weaviate:1.27.0      "/bin/weaviate --hos…"  weaviate    34 seconds ago  Up 33 seconds       

docker-web-1       langgenius/dify-web:1.9.2          "/bin/sh ./entrypoin…"  web       34 seconds ago  Up 33 seconds       3000/tcp

docker-worker-1     langgenius/dify-api:1.9.2          "/bin/bash /entrypoi…"  worker     34 seconds ago  Up 30 seconds       5001/tcp

docker-worker_beat-1   langgenius/dify-api:1.9.2          "/bin/bash /entrypoi…"  worker_beat   34 seconds ago  Up 30 seconds       5001/tcp

(base) liufengxiang@liufengangdeMBP docker % 

# 二、安装效果截图

输入http://localhost/explore/apps，显示如下：

![image-20251022202322395](/Users/liufengxiang/Library/Application Support/typora-user-images/image-20251022202322395.png)