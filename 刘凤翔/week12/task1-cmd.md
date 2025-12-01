Last login: Wed Nov  5 20:55:49 on ttys006
(base) liufengxiang@liufengangdeMBP ~ % pip install -U openai-agents
Looking in indexes: https://mirror.baidu.com/pypi/simple/
ERROR: Could not find a version that satisfies the requirement openai-agents (from versions: none)
ERROR: No matching distribution found for openai-agents
(base) liufengxiang@liufengangdeMBP ~ % conda active
usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...
conda: error: argument COMMAND: invalid choice: 'active' (choose from 'activate', 'clean', 'commands', 'compare', 'config', 'create', 'deactivate', 'env', 'export', 'info', 'init', 'install', 'list', 'notices', 'package', 'content-trust', 'doctor', 'repoquery', 'remove', 'uninstall', 'rename', 'run', 'search', 'update', 'upgrade')
(base) liufengxiang@liufengangdeMBP ~ % conda activate
(base) liufengxiang@liufengangdeMBP ~ % python --version
Python 3.12.7
(base) liufengxiang@liufengangdeMBP ~ % conda env list
# conda environments:
#
                         /Users/liufengxiang/.codegeex/mamba/envs/codegeex-agent
base                  *  /opt/miniconda3
py312                    /opt/miniconda3/envs/py312

(base) liufengxiang@liufengangdeMBP ~ % conda activate py312
(py312) liufengxiang@liufengangdeMBP ~ % pip install fastmcp streamlit fastapi uvicorn requests httpx
Looking in indexes: https://mirror.baidu.com/pypi/simple/
ERROR: Could not find a version that satisfies the requirement fastmcp (from versions: none)
ERROR: No matching distribution found for fastmcp
(py312) liufengxiang@liufengangdeMBP ~ % pip install fastmcp streamlit fastapi uvicorn requests httpx -i https://pypi.tuna.tsinghua.edu.cn/simple/
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
Collecting fastmcp
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9b/4b/7e36db0a90044be181319ff025be7cc57089ddb6ba8f3712dea543b9cf97/fastmcp-2.13.1-py3-none-any.whl (376 kB)
Collecting streamlit
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/60/868371b6482ccd9ef423c6f62650066cf8271fdb2ee84f192695ad6b7a96/streamlit-1.51.0-py3-none-any.whl (10.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.2/10.2 MB 1.8 MB/s eta 0:00:00
Requirement already satisfied: fastapi in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (0.116.1)
Collecting uvicorn
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ee/d9/d88e73ca598f4f6ff671fb5fde8a32925c2e08a637303a1d12883c7305fa/uvicorn-0.38.0-py3-none-any.whl (68 kB)
Requirement already satisfied: requests in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (2.32.5)
Requirement already satisfied: httpx in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (0.28.1)
Collecting authlib>=1.6.5 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f8/aa/5082412d1ee302e9e7d80b6949bc4d2a8fa1149aaab610c5fc24709605d6/authlib-1.6.5-py2.py3-none-any.whl (243 kB)
Collecting cyclopts>=4.0.0 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d2/b2/fabcd6020b63b9d9f7a79cfc61b9c03c4e08ccb54f9cf9db9791be5669ef/cyclopts-4.2.4-py3-none-any.whl (185 kB)
Collecting exceptiongroup>=1.2.2 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/36/f4/c6e662dade71f56cd2f3735141b265c3c79293c109549c1e6933b0651ffc/exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
Collecting jsonschema-path>=0.3.4 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cb/58/3485da8cb93d2f393bce453adeef16896751f14ba3e2024bc21dc9597646/jsonschema_path-0.3.4-py3-none-any.whl (14 kB)
Collecting mcp!=1.21.1,<2.0.0,>=1.19.0 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/67/0f/669ecbe78a0ba192afcc0b026ae62d1005779e91bad27ab9d703401510bf/mcp-1.21.2-py3-none-any.whl (174 kB)
Collecting openapi-pydantic>=0.5.1 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/12/cf/03675d8bd8ecbf4445504d8071adab19f5f993676795708e36402ab38263/openapi_pydantic-0.5.1-py3-none-any.whl (96 kB)
Requirement already satisfied: platformdirs>=4.0.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from fastmcp) (4.3.7)
Collecting py-key-value-aio<0.3.0,>=0.2.8 (from py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cd/5a/e56747d87a97ad2aff0f3700d77f186f0704c90c2da03bfed9e113dae284/py_key_value_aio-0.2.8-py3-none-any.whl (69 kB)
Requirement already satisfied: pydantic>=2.11.7 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pydantic[email]>=2.11.7->fastmcp) (2.11.7)
Collecting pyperclip>=1.9.0 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/df/80/fc9d01d5ed37ba4c42ca2b55b4339ae6e200b456be3a1aaddf4a9fa99b8c/pyperclip-1.11.0-py3-none-any.whl (11 kB)
Collecting python-dotenv>=1.1.0 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/14/1b/a298b06749107c305e1fe0f814c6c74aea7b2f1e10989cb30f544a1b3253/python_dotenv-1.2.1-py3-none-any.whl (21 kB)
Collecting rich>=13.9.4 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/25/7a/b0178788f8dc6cafce37a212c99565fa1fe7872c70c6c9c1e1a372d9d88f/rich-14.2.0-py3-none-any.whl (243 kB)
Collecting websockets>=15.0.1 (from fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3d/69/1a681dd6f02180916f116894181eab8b2e25b31e484c5d0eae637ec01f7c/websockets-15.0.1-cp312-cp312-macosx_11_0_arm64.whl (173 kB)
Requirement already satisfied: anyio>=4.5 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (4.10.0)
Collecting httpx-sse>=0.4 (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d2/fd/6668e5aec43ab844de6fc74927e155a3b37bf40d7c3790e49fc0406b6578/httpx_sse-0.4.3-py3-none-any.whl (9.0 kB)
Requirement already satisfied: jsonschema>=4.20.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (4.25.0)
Collecting pydantic-settings>=2.5.2 (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c1/60/5d4751ba3f4a40a6891f24eec885f51afd78d208498268c734e256fb13c4/pydantic_settings-2.12.0-py3-none-any.whl (51 kB)
Collecting pyjwt>=2.10.1 (from pyjwt[crypto]>=2.10.1->mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/61/ad/689f02752eeec26aed679477e80e632ef1b682313be70793d798c1d5fc8f/PyJWT-2.10.1-py3-none-any.whl (22 kB)
Collecting python-multipart>=0.0.9 (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/45/58/38b5afbc1a800eeea951b9285d3912613f2603bdf897a4ab0f4bd7f405fc/python_multipart-0.0.20-py3-none-any.whl (24 kB)
Collecting sse-starlette>=1.6.1 (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/23/a0/984525d19ca5c8a6c33911a0c164b11490dd0f90ff7fd689f704f84e9a11/sse_starlette-3.0.3-py3-none-any.whl (11 kB)
Requirement already satisfied: starlette>=0.27 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (0.47.2)
Requirement already satisfied: typing-extensions>=4.9.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (4.14.1)
Requirement already satisfied: typing-inspection>=0.4.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (0.4.1)
Collecting beartype>=0.22.2 (from py-key-value-aio<0.3.0,>=0.2.8->py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/c9/ceecc71fe2c9495a1d8e08d44f5f31f5bca1350d5b2e27a4b6265424f59e/beartype-0.22.6-py3-none-any.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 1.4 MB/s eta 0:00:00
Collecting py-key-value-shared==0.2.8 (from py-key-value-aio<0.3.0,>=0.2.8->py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/84/7a/1726ceaa3343874f322dd83c9ec376ad81f533df8422b8b1e1233a59f8ce/py_key_value_shared-0.2.8-py3-none-any.whl (14 kB)
Collecting typing-extensions>=4.9.0 (from mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Collecting diskcache>=5.6.0 (from py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3f/27/4570e78fc0bf5ea0ca45eb1de3818a23787af9b390c0b0a0033a1b8236f9/diskcache-5.6.3-py3-none-any.whl (45 kB)
Collecting pathvalidate>=3.3.1 (from py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9a/70/875f4a23bfc4731703a5835487d0d2fb999031bd415e7d17c0ae615c18b7/pathvalidate-3.3.1-py3-none-any.whl (24 kB)
Collecting keyring>=25.6.0 (from py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/81/db/e655086b7f3a705df045bf0933bdd9c2f79bb3c97bfef1384598bb79a217/keyring-25.7.0-py3-none-any.whl (39 kB)
Collecting cachetools>=6.0.0 (from py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e6/46/eb6eca305c77a4489affe1c5d8f4cae82f285d9addd8de4ec084a7184221/cachetools-6.2.2-py3-none-any.whl (11 kB)
Requirement already satisfied: annotated-types>=0.6.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pydantic>=2.11.7->pydantic[email]>=2.11.7->fastmcp) (0.7.0)
Requirement already satisfied: pydantic-core==2.33.2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pydantic>=2.11.7->pydantic[email]>=2.11.7->fastmcp) (2.33.2)
Collecting altair!=5.4.0,!=5.4.1,<6,>=4.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/aa/f3/0b6ced594e51cc95d8c1fc1640d3623770d01e4969d29c0bd09945fafefa/altair-5.5.0-py3-none-any.whl (731 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.2/731.2 kB 1.3 MB/s eta 0:00:00
Collecting blinker<2,>=1.5.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/10/cb/f2ad4230dc2eb1a74edf38f1a38b9b52277f75bef262d8908e60d957e13c/blinker-1.9.0-py3-none-any.whl (8.5 kB)
Collecting click<9,>=7.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/78/01c019cdb5d6498122777c1a43056ebb3ebfeef2076d9d026bfe15583b2b/click-8.3.1-py3-none-any.whl (108 kB)
Requirement already satisfied: numpy<3,>=1.23 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (1.26.4)
Requirement already satisfied: packaging<26,>=20 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (25.0)
Requirement already satisfied: pandas<3,>=1.4.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (2.2.2)
Requirement already satisfied: pillow<13,>=7.1.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (11.3.0)
Requirement already satisfied: protobuf<7,>=3.20 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (6.31.1)
Requirement already satisfied: pyarrow<22,>=7.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (21.0.0)
Collecting tenacity<10,>=8.1.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e5/30/643397144bfbfec6f6ef821f36f33e57d35946c44a2352d3c9f0ae847619/tenacity-9.1.2-py3-none-any.whl (28 kB)
Collecting toml<2,>=0.10.1 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/44/6f/7120676b6d73228c96e17f1f794d8ab046fc910d781c8d151120c3f1569e/toml-0.10.2-py2.py3-none-any.whl (16 kB)
Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/01/61/d4b89fec821f72385526e1b9d9a3a0385dda4a72b206d28049e2c7cd39b8/gitpython-3.1.45-py3-none-any.whl (208 kB)
Collecting pydeck<1,>=0.8.0b4 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ab/4c/b888e6cf58bd9db9c93f40d1c6be8283ff49d88919231afe93a6bcf61626/pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 1.6 MB/s eta 0:00:00
Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from streamlit) (6.5.1)
Requirement already satisfied: charset_normalizer<4,>=2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from requests) (3.4.3)
Requirement already satisfied: idna<4,>=2.5 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from requests) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from requests) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from requests) (2025.10.5)
Requirement already satisfied: jinja2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.1.6)
Collecting narwhals>=1.14.2 (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0b/9a/c6f79de7ba3a0a8473129936b7b90aa461d3d46fec6f1627672b1dccf4e9/narwhals-2.12.0-py3-none-any.whl (425 kB)
Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a0/61/5c78b91c3143ed5c14207f463aecfc8f9dbb5092fb2869baf37c273b2705/gitdb-4.0.12-py3-none-any.whl (62 kB)
Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/04/be/d09147ad1ec7934636ad912901c5fd7667e1c858e19d355237db0d0cd5e4/smmap-5.0.2-py3-none-any.whl (24 kB)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)
Requirement already satisfied: sniffio>=1.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from anyio>=4.5->mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (1.3.1)
Requirement already satisfied: h11>=0.8 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from uvicorn) (0.16.0)
Requirement already satisfied: httpcore==1.* in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from httpx) (1.0.9)
Requirement already satisfied: cryptography in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from authlib>=1.6.5->fastmcp) (46.0.1)
Requirement already satisfied: attrs>=23.1.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from cyclopts>=4.0.0->fastmcp) (25.3.0)
Collecting docstring-parser<4.0,>=0.15 (from cyclopts>=4.0.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/55/e2/2537ebcff11c1ee1ff17d8d0b6f4db75873e3b0fb32c2d4a2ee31ecb310a/docstring_parser-0.17.0-py3-none-any.whl (36 kB)
Collecting rich-rst<2.0.0,>=1.3.1 (from cyclopts>=4.0.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/13/2f/b4530fbf948867702d0a3f27de4a6aab1d156f406d72852ab902c4d04de9/rich_rst-1.3.2-py3-none-any.whl (12 kB)
Collecting docutils (from rich-rst<2.0.0,>=1.3.1->cyclopts>=4.0.0->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/11/a8/c6a4b901d17399c77cd81fb001ce8961e9f5e04d3daf27e8925cb012e163/docutils-0.22.3-py3-none-any.whl (633 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 633.0/633.0 kB 1.4 MB/s eta 0:00:00
Requirement already satisfied: MarkupSafe>=2.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.0.2)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (2023.7.1)
Requirement already satisfied: referencing>=0.28.4 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (0.30.2)
Requirement already satisfied: rpds-py>=0.7.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp!=1.21.1,<2.0.0,>=1.19.0->fastmcp) (0.22.3)
Requirement already satisfied: PyYAML>=5.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema-path>=0.3.4->fastmcp) (6.0.2)
Collecting pathable<0.5.0,>=0.4.1 (from jsonschema-path>=0.3.4->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7d/eb/b6260b31b1a96386c0a880edebe26f89669098acea8e0318bff6adb378fd/pathable-0.4.4-py3-none-any.whl (9.6 kB)
Collecting jaraco.classes (from keyring>=25.6.0->py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7f/66/b15ce62552d84bbfcec9a4873ab79d993a1dd4edb922cbfccae192bd5b5f/jaraco.classes-3.4.0-py3-none-any.whl (6.8 kB)
Collecting jaraco.functools (from keyring>=25.6.0->py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b4/09/726f168acad366b11e420df31bf1c702a54d373a83f968d94141a8c3fde0/jaraco_functools-4.3.0-py3-none-any.whl (10 kB)
Collecting jaraco.context (from keyring>=25.6.0->py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ff/db/0c52c4cf5e4bd9f5d7135ec7669a3a767af21b3a308e1ed3674881e52b62/jaraco.context-6.0.1-py3-none-any.whl (6.8 kB)
Collecting email-validator>=2.0.0 (from pydantic[email]>=2.11.7->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/de/15/545e2b6cf2e3be84bc1ed85613edd75b8aea69807a71c26f4ca6a9258e82/email_validator-2.3.0-py3-none-any.whl (35 kB)
Collecting dnspython>=2.0.0 (from email-validator>=2.0.0->pydantic[email]>=2.11.7->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ba/5a/18ad964b0086c6e62e2e7500f7edc89e3faa45033c71c1893d34eed2b2de/dnspython-2.8.0-py3-none-any.whl (331 kB)
Requirement already satisfied: cffi>=2.0.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from cryptography->authlib>=1.6.5->fastmcp) (2.0.0)
Requirement already satisfied: pycparser in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from cffi>=2.0.0->cryptography->authlib>=1.6.5->fastmcp) (2.23)
Requirement already satisfied: six>=1.5 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)
Collecting markdown-it-py>=2.2.0 (from rich>=13.9.4->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/94/54/e7d793b573f298e1c9013b8c4dade17d481164aa517d1d7148619c2cedbf/markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from rich>=13.9.4->fastmcp) (2.19.1)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=13.9.4->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Collecting more-itertools (from jaraco.classes->keyring>=25.6.0->py-key-value-aio[disk,keyring,memory]<0.3.0,>=0.2.8->fastmcp)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a4/8e/469e5a4a2f5855992e425f3cb33804cc07bf18d48f2db061aec61ce50270/more_itertools-10.8.0-py3-none-any.whl (69 kB)
Installing collected packages: pyperclip, websockets, typing-extensions, toml, tenacity, smmap, python-multipart, python-dotenv, pyjwt, pathvalidate, pathable, narwhals, more-itertools, mdurl, jaraco.context, httpx-sse, docutils, docstring-parser, dnspython, diskcache, click, cachetools, blinker, beartype, uvicorn, pydeck, py-key-value-shared, markdown-it-py, jsonschema-path, jaraco.functools, jaraco.classes, gitdb, exceptiongroup, email-validator, sse-starlette, rich, py-key-value-aio, keyring, gitpython, authlib, rich-rst, pydantic-settings, openapi-pydantic, altair, streamlit, mcp, cyclopts, fastmcp
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.14.1
    Uninstalling typing_extensions-4.14.1:
      Successfully uninstalled typing_extensions-4.14.1
Successfully installed altair-5.5.0 authlib-1.6.5 beartype-0.22.6 blinker-1.9.0 cachetools-6.2.2 click-8.3.1 cyclopts-4.2.4 diskcache-5.6.3 dnspython-2.8.0 docstring-parser-0.17.0 docutils-0.22.3 email-validator-2.3.0 exceptiongroup-1.3.0 fastmcp-2.13.1 gitdb-4.0.12 gitpython-3.1.45 httpx-sse-0.4.3 jaraco.classes-3.4.0 jaraco.context-6.0.1 jaraco.functools-4.3.0 jsonschema-path-0.3.4 keyring-25.7.0 markdown-it-py-4.0.0 mcp-1.21.2 mdurl-0.1.2 more-itertools-10.8.0 narwhals-2.12.0 openapi-pydantic-0.5.1 pathable-0.4.4 pathvalidate-3.3.1 py-key-value-aio-0.2.8 py-key-value-shared-0.2.8 pydantic-settings-2.12.0 pydeck-0.9.1 pyjwt-2.10.1 pyperclip-1.11.0 python-dotenv-1.2.1 python-multipart-0.0.20 rich-14.2.0 rich-rst-1.3.2 smmap-5.0.2 sse-starlette-3.0.3 streamlit-1.51.0 tenacity-9.1.2 toml-0.10.2 typing-extensions-4.15.0 uvicorn-0.38.0 websockets-15.0.1
(py312) liufengxiang@liufengangdeMBP ~ % pip install git+https://github.com/openai/openai-agents-python.git
Looking in indexes: https://mirror.baidu.com/pypi/simple/
Collecting git+https://github.com/openai/openai-agents-python.git
  Cloning https://github.com/openai/openai-agents-python.git to /private/var/folders/3j/ckt4dhp11jq04g_wf5plnmpc0000gn/T/pip-req-build-re1dqt49
  Running command git clone --filter=blob:none --quiet https://github.com/openai/openai-agents-python.git /private/var/folders/3j/ckt4dhp11jq04g_wf5plnmpc0000gn/T/pip-req-build-re1dqt49
  Resolved https://github.com/openai/openai-agents-python.git to commit 0dae2f0f1f2c5ac1a49fd54db56c60d69b93674c
  Installing build dependencies ... error
  error: subprocess-exited-with-error
  
  × pip subprocess to install build dependencies did not run successfully.
  │ exit code: 1
  ╰─> [3 lines of output]
      Looking in indexes: https://mirror.baidu.com/pypi/simple/
      ERROR: Could not find a version that satisfies the requirement hatchling (from versions: none)
      ERROR: No matching distribution found for hatchling
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× pip subprocess to install build dependencies did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
(py312) liufengxiang@liufengangdeMBP ~ % pip install git+https://github.com/openai/openai-agents-python.git -i https://pypi.tuna.tsinghua.edu.cn/simple
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting git+https://github.com/openai/openai-agents-python.git
  Cloning https://github.com/openai/openai-agents-python.git to /private/var/folders/3j/ckt4dhp11jq04g_wf5plnmpc0000gn/T/pip-req-build-nxcrq4jo
  Running command git clone --filter=blob:none --quiet https://github.com/openai/openai-agents-python.git /private/var/folders/3j/ckt4dhp11jq04g_wf5plnmpc0000gn/T/pip-req-build-nxcrq4jo
  Resolved https://github.com/openai/openai-agents-python.git to commit 0dae2f0f1f2c5ac1a49fd54db56c60d69b93674c
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting griffe<2,>=1.5.6 (from openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9c/83/3b1d03d36f224edded98e9affd0467630fc09d766c0e56fb1498cbb04a9b/griffe-1.15.0-py3-none-any.whl (150 kB)
Requirement already satisfied: mcp<2,>=1.11.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from openai-agents==0.6.1) (1.21.2)
Collecting openai<3,>=2.8.0 (from openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/55/4f/dbc0c124c40cb390508a82770fb9f6e3ed162560181a85089191a851c59a/openai-2.8.1-py3-none-any.whl (1.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 9.2 MB/s eta 0:00:00
Collecting pydantic<3,>=2.12.3 (from openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/82/2f/e68750da9b04856e2a7ec56fc6f034a5a79775e9b9a81882252789873798/pydantic-2.12.4-py3-none-any.whl (463 kB)
Requirement already satisfied: requests<3,>=2.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from openai-agents==0.6.1) (2.32.5)
Collecting types-requests<3,>=2.0 (from openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/20/9a227ea57c1285986c4cf78400d0a91615d25b24e257fd9e2969606bdfae/types_requests-2.32.4.20250913-py3-none-any.whl (20 kB)
Requirement already satisfied: typing-extensions<5,>=4.12.2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from openai-agents==0.6.1) (4.15.0)
Collecting colorama>=0.4 (from griffe<2,>=1.5.6->openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d1/d6/3965ed04c63042e047cb6a3e6ed1a63a35087b6a609aa3a15ed8ac56c221/colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Requirement already satisfied: anyio>=4.5 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (4.10.0)
Requirement already satisfied: httpx-sse>=0.4 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (0.4.3)
Requirement already satisfied: httpx>=0.27.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (0.28.1)
Requirement already satisfied: jsonschema>=4.20.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (4.25.0)
Requirement already satisfied: pydantic-settings>=2.5.2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (2.12.0)
Requirement already satisfied: pyjwt>=2.10.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pyjwt[crypto]>=2.10.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (2.10.1)
Requirement already satisfied: python-multipart>=0.0.9 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (0.0.20)
Requirement already satisfied: sse-starlette>=1.6.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (3.0.3)
Requirement already satisfied: starlette>=0.27 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (0.47.2)
Requirement already satisfied: typing-inspection>=0.4.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (0.4.1)
Requirement already satisfied: uvicorn>=0.31.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from mcp<2,>=1.11.0->openai-agents==0.6.1) (0.38.0)
Collecting distro<2,>=1.7.0 (from openai<3,>=2.8.0->openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/12/b3/231ffd4ab1fc9d679809f356cebee130ac7daa00d6d6f3206dd4fd137e9e/distro-1.9.0-py3-none-any.whl (20 kB)
Collecting jiter<1,>=0.10.0 (from openai<3,>=2.8.0->openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/6e/e8efa0e78de00db0aee82c0cf9e8b3f2027efd7f8a71f859d8f4be8e98ef/jiter-0.12.0-cp312-cp312-macosx_11_0_arm64.whl (319 kB)
Requirement already satisfied: sniffio in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from openai<3,>=2.8.0->openai-agents==0.6.1) (1.3.1)
Requirement already satisfied: tqdm>4 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from openai<3,>=2.8.0->openai-agents==0.6.1) (4.67.1)
Requirement already satisfied: idna>=2.8 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from anyio>=4.5->mcp<2,>=1.11.0->openai-agents==0.6.1) (3.10)
Requirement already satisfied: certifi in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from httpx>=0.27.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (2025.10.5)
Requirement already satisfied: httpcore==1.* in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from httpx>=0.27.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (1.0.9)
Requirement already satisfied: h11>=0.16 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from httpcore==1.*->httpx>=0.27.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (0.16.0)
Requirement already satisfied: annotated-types>=0.6.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pydantic<3,>=2.12.3->openai-agents==0.6.1) (0.7.0)
Collecting pydantic-core==2.41.5 (from pydantic<3,>=2.12.3->openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/aa/32/9c2e8ccb57c01111e0fd091f236c7b371c1bccea0fa85247ac55b1e2b6b6/pydantic_core-2.41.5-cp312-cp312-macosx_11_0_arm64.whl (1.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 10.9 MB/s eta 0:00:00
Collecting typing-inspection>=0.4.1 (from mcp<2,>=1.11.0->openai-agents==0.6.1)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dc/9b/47798a6c91d8bdb567fe2698fe81e0c6b7cb7ef4d13da4114b41d239f65d/typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Requirement already satisfied: charset_normalizer<4,>=2 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from requests<3,>=2.0->openai-agents==0.6.1) (3.4.3)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from requests<3,>=2.0->openai-agents==0.6.1) (2.5.0)
Requirement already satisfied: attrs>=22.2.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp<2,>=1.11.0->openai-agents==0.6.1) (25.3.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp<2,>=1.11.0->openai-agents==0.6.1) (2023.7.1)
Requirement already satisfied: referencing>=0.28.4 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp<2,>=1.11.0->openai-agents==0.6.1) (0.30.2)
Requirement already satisfied: rpds-py>=0.7.1 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from jsonschema>=4.20.0->mcp<2,>=1.11.0->openai-agents==0.6.1) (0.22.3)
Requirement already satisfied: python-dotenv>=0.21.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pydantic-settings>=2.5.2->mcp<2,>=1.11.0->openai-agents==0.6.1) (1.2.1)
Requirement already satisfied: cryptography>=3.4.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from pyjwt[crypto]>=2.10.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (46.0.1)
Requirement already satisfied: cffi>=2.0.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from cryptography>=3.4.0->pyjwt[crypto]>=2.10.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (2.0.0)
Requirement already satisfied: pycparser in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from cffi>=2.0.0->cryptography>=3.4.0->pyjwt[crypto]>=2.10.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (2.23)
Requirement already satisfied: click>=7.0 in /opt/miniconda3/envs/py312/lib/python3.12/site-packages (from uvicorn>=0.31.1->mcp<2,>=1.11.0->openai-agents==0.6.1) (8.3.1)
Building wheels for collected packages: openai-agents
  Building wheel for openai-agents (pyproject.toml) ... done
  Created wheel for openai-agents: filename=openai_agents-0.6.1-py3-none-any.whl size=237610 sha256=e6b643b9819aac8d464e45b4df5ce133af7c89a609076711755a4e7af6e4587f
  Stored in directory: /private/var/folders/3j/ckt4dhp11jq04g_wf5plnmpc0000gn/T/pip-ephem-wheel-cache-uglt8fse/wheels/93/09/1c/fbac0047d4f0e7d3dda2daa184dff45a5b38758ea93d5d8485
Successfully built openai-agents
Installing collected packages: typing-inspection, types-requests, pydantic-core, jiter, distro, colorama, pydantic, griffe, openai, openai-agents
  Attempting uninstall: typing-inspection
    Found existing installation: typing-inspection 0.4.1
    Uninstalling typing-inspection-0.4.1:
      Successfully uninstalled typing-inspection-0.4.1
  Attempting uninstall: pydantic-core
    Found existing installation: pydantic_core 2.33.2
    Uninstalling pydantic_core-2.33.2:
      Successfully uninstalled pydantic_core-2.33.2
  Attempting uninstall: pydantic
    Found existing installation: pydantic 2.11.7
    Uninstalling pydantic-2.11.7:
      Successfully uninstalled pydantic-2.11.7
Successfully installed colorama-0.4.6 distro-1.9.0 griffe-1.15.0 jiter-0.12.0 openai-2.8.1 openai-agents-0.6.1 pydantic-2.12.4 pydantic-core-2.41.5 types-requests-2.32.4.20250913 typing-inspection-0.4.2
(py312) liufengxiang@liufengangdeMBP ~ % # 检查关键包是否安装成功
python -c "
import fastmcp; print('✅ fastmcp 安装成功')
import streamlit; print('✅ streamlit 安装成功')
import fastapi; print('✅ fastapi 安装成功')
import requests; print('✅ requests 安装成功')
try:
    import agents; print('✅ openai-agents 安装成功')
except ImportError:
    print('⚠<fe0f> openai-agents 未安装，部分功能可能受限')
"
zsh: command not found: #
✅ fastmcp 安装成功
✅ streamlit 安装成功
✅ fastapi 安装成功
✅ requests 安装成功
✅ openai-agents 安装成功
(py312) liufengxiang@liufengangdeMBP ~ % python mcp_server_main.py
python: can't open file '/Users/liufengxiang/mcp_server_main.py': [Errno 2] No such file or directory
(py312) liufengxiang@liufengangdeMBP ~ % ls
Applications			Public
CodeGeeXProjects		dify_test
Desktop				miniconda3
Documents			models
Downloads			player.list
Library				requirements.txt
Movies				stable-diffusion-webui
Music				stable-diffusion-webui_坏的
Pictures			venv
(py312) liufengxiang@liufengangdeMBP ~ % cd /Users/liufengxiang/Documents/锦汐· 沐瑶/第12周：Agent搭建与MCP应用/Week12/Week12/4-项目案例-企业职能助手/mcp_server/mcp_server_main.py
cd: not a directory: /Users/liufengxiang/Documents/锦汐·沐瑶/第12周：Agent搭建与MCP应用/Week12/Week12/4-项目案例-企业职能助手/mcp_server/mcp_server_main.py
(py312) liufengxiang@liufengangdeMBP ~ % ls
Applications			Public
CodeGeeXProjects		dify_test
Desktop				miniconda3
Documents			models
Downloads			player.list
Library				requirements.txt
Movies				stable-diffusion-webui
Music				stable-diffusion-webui_坏的
Pictures			venv
(py312) liufengxiang@liufengangdeMBP ~ % cd /Users/liufengxiang/Documents/锦汐· 沐瑶/第12周：Agent搭建与MCP应用/Week12/Week12/4-项目案例-企业职能助手/mcp_server 
(py312) liufengxiang@liufengangdeMBP mcp_server % ls
__pycache__		mcp_server_main.py	tool.py
mcp_client_test1.py	news.py
mcp_client_test2.py	saying.py
(py312) liufengxiang@liufengangdeMBP mcp_server % python mcp_server_main.py
Available tools: ['get_today_daily_news', 'get_douyin_hot_news', 'get_github_hot_news', 'get_toutiao_hot_news', 'get_sports_news', 'get_today_familous_saying', 'get_today_motivation_saying', 'get_today_working_saying', 'get_city_weather', 'get_address_detail', 'get_tel_info', 'get_scenic_info', 'get_flower_info', 'get_rate_transform']


╭──────────────────────────────────────────────────────────────────────────────╮
│                                                                              │
│                         ▄▀▀ ▄▀█ █▀▀ ▀█▀ █▀▄▀█ █▀▀ █▀█                        │
│                         █▀  █▀█ ▄▄█  █  █ ▀ █ █▄▄ █▀▀                        │
│                                                                              │
│                                FastMCP 2.13.1                                │
│                                                                              │
│                                                                              │
│                  🖥  Server name: MCP-Server                                  │
│                                                                              │
│                  📦 Transport:   SSE                                         │
│                  🔗 Server URL:  http://127.0.0.1:8900/sse                   │
│                                                                              │
│                  📚 Docs:        https://gofastmcp.com                       │
│                  🚀 Hosting:     https://fastmcp.cloud                       │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯


[11/20/25 20:54:21] INFO     Starting MCP server 'MCP-Server'     server.py:2055
                             with transport 'sse' on                            
                             http://127.0.0.1:8900/sse                          
INFO:     Started server process [86653]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8900 (Press CTRL+C to quit)
INFO:     127.0.0.1:64194 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:64468 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:64470 - "POST /messages/?session_id=fa401b3ebf9f44d491368c430474a616 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:64472 - "POST /messages/?session_id=fa401b3ebf9f44d491368c430474a616 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:64474 - "POST /messages/?session_id=fa401b3ebf9f44d491368c430474a616 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:64476 - "POST /messages/?session_id=fa401b3ebf9f44d491368c430474a616 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51172 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:51174 - "POST /messages/?session_id=38dc5751307c461281ba81ede9bd0ab5 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51176 - "POST /messages/?session_id=38dc5751307c461281ba81ede9bd0ab5 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51178 - "POST /messages/?session_id=38dc5751307c461281ba81ede9bd0ab5 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51206 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:51208 - "POST /messages/?session_id=04fad6dc59ed4eaca6dadb7545d45c17 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51210 - "POST /messages/?session_id=04fad6dc59ed4eaca6dadb7545d45c17 HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51212 - "POST /messages/?session_id=04fad6dc59ed4eaca6dadb7545d45c17 HTTP/1.1" 202 Accepted

