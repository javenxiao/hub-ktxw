Last login: Wed Oct 22 19:20:33 on ttys005
(base) liufengxiang@liufengangdeMBP ~ % conda dev list
usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...
conda: error: argument COMMAND: invalid choice: 'dev' (choose from 'activate', 'clean', 'commands', 'compare', 'config', 'create', 'deactivate', 'env', 'export', 'info', 'init', 'install', 'list', 'notices', 'package', 'content-trust', 'doctor', 'repoquery', 'remove', 'uninstall', 'rename', 'run', 'search', 'update', 'upgrade')
(base) liufengxiang@liufengangdeMBP ~ % conda env list
# conda environments:
#
                         /Users/liufengxiang/.codegeex/mamba/envs/codegeex-agent
base                  *  /opt/miniconda3
py312                    /opt/miniconda3/envs/py312

(base) liufengxiang@liufengangdeMBP ~ % conda activate py312
(py312) liufengxiang@liufengangdeMBP ~ % conda install jupyter
Retrieving notices: ...working... done
Channels:
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /opt/miniconda3/envs/py312

  added / updated specs:
    - jupyter


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    anyio-4.10.0               |  py312hca03da5_0         287 KB
    appnope-0.1.4              |  py312hca03da5_0          15 KB
    argon2-cffi-21.3.0         |     pyhd3eb1b0_0          15 KB
    argon2-cffi-bindings-25.1.0|  py312h254cc4a_0          31 KB
    asttokens-3.0.0            |  py312hca03da5_0          48 KB
    async-lru-2.0.5            |  py312hca03da5_0          21 KB
    attrs-25.4.0               |  py312hca03da5_1         171 KB
    babel-2.16.0               |  py312hca03da5_0         6.8 MB
    beautifulsoup4-4.13.5      |  py312hca03da5_0         257 KB
    bleach-6.2.0               |  py312hca03da5_0         360 KB
    brotlicffi-1.0.9.2         |  py312h313beb8_1         373 KB
    ca-certificates-2025.11.4  |       hca03da5_0         128 KB
    certifi-2025.10.5          |  py312hca03da5_0         157 KB
    cffi-2.0.0                 |  py312h73c2a22_1         279 KB
    charset-normalizer-3.4.4   |  py312hca03da5_0          97 KB
    comm-0.2.1                 |  py312hca03da5_0          18 KB
    debugpy-1.8.16             |  py312h0962b89_0         2.5 MB
    decorator-5.2.1            |  py312hca03da5_0          42 KB
    defusedxml-0.7.1           |     pyhd3eb1b0_0          23 KB
    executing-2.2.1            |  py312hca03da5_0         334 KB
    h11-0.16.0                 |  py312hca03da5_1          62 KB
    httpcore-1.0.9             |  py312hca03da5_0         125 KB
    httpx-0.28.1               |  py312hca03da5_1         210 KB
    idna-3.11                  |  py312hca03da5_0         198 KB
    ipykernel-6.30.1           |  py312hca03da5_0         243 KB
    ipython-9.5.0              |  py312hca03da5_0         1.1 MB
    ipython_pygments_lexers-1.1.1|  py312hca03da5_0          19 KB
    ipywidgets-8.1.7           |  py312hca03da5_0         240 KB
    jedi-0.19.2                |  py312hca03da5_0         1.2 MB
    jinja2-3.1.6               |  py312hca03da5_0         360 KB
    json5-0.9.25               |  py312hca03da5_0          54 KB
    jsonschema-4.25.0          |  py312hca03da5_0         199 KB
    jsonschema-specifications-2023.7.1|  py312hca03da5_0          16 KB
    jupyter-1.1.1              |  py312hca03da5_0           9 KB
    jupyter-lsp-2.2.5          |  py312hca03da5_0         115 KB
    jupyter_client-8.6.3       |  py312hca03da5_0         237 KB
    jupyter_console-6.6.3      |  py312hca03da5_1          53 KB
    jupyter_core-5.8.1         |  py312hca03da5_0          97 KB
    jupyter_events-0.12.0      |  py312hca03da5_0          43 KB
    jupyter_server-2.16.0      |  py312hca03da5_0         605 KB
    jupyter_server_terminals-0.5.3|  py312hca03da5_0          30 KB
    jupyterlab-4.4.7           |  py312hca03da5_0         8.6 MB
    jupyterlab_pygments-0.3.0  |  py312hca03da5_0          20 KB
    jupyterlab_server-2.28.0   |  py312hca03da5_0         118 KB
    jupyterlab_widgets-3.0.15  |  py312hca03da5_0         199 KB
    libsodium-1.0.20           |       h897f8a9_0         224 KB
    markupsafe-3.0.2           |  py312h80987f9_0          27 KB
    matplotlib-inline-0.1.7    |  py312hca03da5_0          19 KB
    mistune-3.1.2              |  py312hca03da5_0         147 KB
    nbclient-0.10.2            |  py312hca03da5_0          55 KB
    nbconvert-7.16.6           |  py312hca03da5_0           9 KB
    nbconvert-core-7.16.6      |  py312hca03da5_0         516 KB
    nbconvert-pandoc-7.16.6    |  py312hca03da5_0           9 KB
    nbformat-5.10.4            |  py312hca03da5_0         158 KB
    nest-asyncio-1.6.0         |  py312hca03da5_0          17 KB
    notebook-7.4.5             |  py312hca03da5_0         6.2 MB
    notebook-shim-0.2.4        |  py312hca03da5_0          27 KB
    openssl-3.0.18             |       h9b4081a_0         3.1 MB
    overrides-7.4.0            |  py312hca03da5_0          36 KB
    packaging-25.0             |  py312hca03da5_1         188 KB
    pandoc-3.8                 |       hca03da5_0        26.4 MB
    pandocfilters-1.5.1        |  py312hca03da5_0          17 KB
    parso-0.8.4                |  py312hca03da5_0         239 KB
    pexpect-4.9.0              |  py312hca03da5_0         158 KB
    platformdirs-4.3.7         |  py312hca03da5_0          42 KB
    prometheus_client-0.21.1   |  py312hca03da5_0         153 KB
    prompt-toolkit-3.0.52      |  py312hca03da5_1         703 KB
    prompt_toolkit-3.0.52      |       hd3eb1b0_1           5 KB
    psutil-7.0.0               |  py312haa24f5a_1         539 KB
    ptyprocess-0.7.0           |     pyhd3eb1b0_2          17 KB
    pure_eval-0.2.3            |  py312hca03da5_0          34 KB
    pycparser-2.23             |  py312hca03da5_0         259 KB
    pygments-2.19.1            |  py312hca03da5_0         2.2 MB
    pysocks-1.7.1              |  py312hca03da5_1          35 KB
    python-dateutil-2.9.0post0 |  py312hca03da5_2         320 KB
    python-fastjsonschema-2.20.0|  py312hca03da5_0         262 KB
    python-json-logger-3.2.1   |  py312hca03da5_0          29 KB
    pyyaml-6.0.2               |  py312h80987f9_0         200 KB
    pyzmq-27.1.0               |  py312h854a7ef_0         347 KB
    qtconsole-5.7.0            |  py312hca03da5_0         253 KB
    qtpy-2.4.3                 |  py312hca03da5_0         129 KB
    referencing-0.30.2         |  py312hca03da5_0          73 KB
    requests-2.32.5            |  py312hca03da5_0         173 KB
    rfc3339-validator-0.1.4    |  py312hca03da5_0           9 KB
    rfc3986-validator-0.1.1    |  py312hca03da5_0           9 KB
    rpds-py-0.22.3             |  py312h2aea54e_0         341 KB
    send2trash-1.8.2           |  py312hca03da5_1          33 KB
    six-1.17.0                 |  py312hca03da5_0          38 KB
    sniffio-1.3.0              |  py312hca03da5_0          18 KB
    soupsieve-2.5              |  py312hca03da5_0          84 KB
    stack_data-0.6.3           |  py312hca03da5_0          65 KB
    terminado-0.18.1           |  py312hca03da5_0          32 KB
    tinycss2-1.4.0             |  py312hca03da5_0         110 KB
    tornado-6.5.1              |  py312h80987f9_0         877 KB
    traitlets-5.14.3           |  py312hca03da5_0         216 KB
    typing-extensions-4.15.0   |  py312hca03da5_0          12 KB
    typing_extensions-4.15.0   |  py312hca03da5_0         100 KB
    urllib3-2.5.0              |  py312hca03da5_0         353 KB
    wcwidth-0.2.13             |  py312hca03da5_0          74 KB
    webencodings-0.5.1         |  py312hca03da5_2          25 KB
    websocket-client-1.8.0     |  py312hca03da5_0         115 KB
    widgetsnbextension-4.0.14  |  py312hca03da5_0         867 KB
    yaml-0.2.5                 |       h1a28f6b_0          71 KB
    zeromq-4.3.5               |       h2c7f8f0_1         277 KB
    ------------------------------------------------------------
                                           Total:        72.9 MB

The following NEW packages will be INSTALLED:

  anyio              pkgs/main/osx-arm64::anyio-4.10.0-py312hca03da5_0 
  appnope            pkgs/main/osx-arm64::appnope-0.1.4-py312hca03da5_0 
  argon2-cffi        pkgs/main/noarch::argon2-cffi-21.3.0-pyhd3eb1b0_0 
  argon2-cffi-bindi~ pkgs/main/osx-arm64::argon2-cffi-bindings-25.1.0-py312h254cc4a_0 
  asttokens          pkgs/main/osx-arm64::asttokens-3.0.0-py312hca03da5_0 
  async-lru          pkgs/main/osx-arm64::async-lru-2.0.5-py312hca03da5_0 
  attrs              pkgs/main/osx-arm64::attrs-25.4.0-py312hca03da5_1 
  babel              pkgs/main/osx-arm64::babel-2.16.0-py312hca03da5_0 
  beautifulsoup4     pkgs/main/osx-arm64::beautifulsoup4-4.13.5-py312hca03da5_0 
  bleach             pkgs/main/osx-arm64::bleach-6.2.0-py312hca03da5_0 
  brotlicffi         pkgs/main/osx-arm64::brotlicffi-1.0.9.2-py312h313beb8_1 
  certifi            pkgs/main/osx-arm64::certifi-2025.10.5-py312hca03da5_0 
  cffi               pkgs/main/osx-arm64::cffi-2.0.0-py312h73c2a22_1 
  charset-normalizer pkgs/main/osx-arm64::charset-normalizer-3.4.4-py312hca03da5_0 
  comm               pkgs/main/osx-arm64::comm-0.2.1-py312hca03da5_0 
  debugpy            pkgs/main/osx-arm64::debugpy-1.8.16-py312h0962b89_0 
  decorator          pkgs/main/osx-arm64::decorator-5.2.1-py312hca03da5_0 
  defusedxml         pkgs/main/noarch::defusedxml-0.7.1-pyhd3eb1b0_0 
  executing          pkgs/main/osx-arm64::executing-2.2.1-py312hca03da5_0 
  h11                pkgs/main/osx-arm64::h11-0.16.0-py312hca03da5_1 
  httpcore           pkgs/main/osx-arm64::httpcore-1.0.9-py312hca03da5_0 
  httpx              pkgs/main/osx-arm64::httpx-0.28.1-py312hca03da5_1 
  idna               pkgs/main/osx-arm64::idna-3.11-py312hca03da5_0 
  ipykernel          pkgs/main/osx-arm64::ipykernel-6.30.1-py312hca03da5_0 
  ipython            pkgs/main/osx-arm64::ipython-9.5.0-py312hca03da5_0 
  ipython_pygments_~ pkgs/main/osx-arm64::ipython_pygments_lexers-1.1.1-py312hca03da5_0 
  ipywidgets         pkgs/main/osx-arm64::ipywidgets-8.1.7-py312hca03da5_0 
  jedi               pkgs/main/osx-arm64::jedi-0.19.2-py312hca03da5_0 
  jinja2             pkgs/main/osx-arm64::jinja2-3.1.6-py312hca03da5_0 
  json5              pkgs/main/osx-arm64::json5-0.9.25-py312hca03da5_0 
  jsonschema         pkgs/main/osx-arm64::jsonschema-4.25.0-py312hca03da5_0 
  jsonschema-specif~ pkgs/main/osx-arm64::jsonschema-specifications-2023.7.1-py312hca03da5_0 
  jupyter            pkgs/main/osx-arm64::jupyter-1.1.1-py312hca03da5_0 
  jupyter-lsp        pkgs/main/osx-arm64::jupyter-lsp-2.2.5-py312hca03da5_0 
  jupyter_client     pkgs/main/osx-arm64::jupyter_client-8.6.3-py312hca03da5_0 
  jupyter_console    pkgs/main/osx-arm64::jupyter_console-6.6.3-py312hca03da5_1 
  jupyter_core       pkgs/main/osx-arm64::jupyter_core-5.8.1-py312hca03da5_0 
  jupyter_events     pkgs/main/osx-arm64::jupyter_events-0.12.0-py312hca03da5_0 
  jupyter_server     pkgs/main/osx-arm64::jupyter_server-2.16.0-py312hca03da5_0 
  jupyter_server_te~ pkgs/main/osx-arm64::jupyter_server_terminals-0.5.3-py312hca03da5_0 
  jupyterlab         pkgs/main/osx-arm64::jupyterlab-4.4.7-py312hca03da5_0 
  jupyterlab_pygmen~ pkgs/main/osx-arm64::jupyterlab_pygments-0.3.0-py312hca03da5_0 
  jupyterlab_server  pkgs/main/osx-arm64::jupyterlab_server-2.28.0-py312hca03da5_0 
  jupyterlab_widgets pkgs/main/osx-arm64::jupyterlab_widgets-3.0.15-py312hca03da5_0 
  libiconv           pkgs/main/osx-arm64::libiconv-1.16-h80987f9_3 
  libsodium          pkgs/main/osx-arm64::libsodium-1.0.20-h897f8a9_0 
  markupsafe         pkgs/main/osx-arm64::markupsafe-3.0.2-py312h80987f9_0 
  matplotlib-inline  pkgs/main/osx-arm64::matplotlib-inline-0.1.7-py312hca03da5_0 
  mistune            pkgs/main/osx-arm64::mistune-3.1.2-py312hca03da5_0 
  nbclient           pkgs/main/osx-arm64::nbclient-0.10.2-py312hca03da5_0 
  nbconvert          pkgs/main/osx-arm64::nbconvert-7.16.6-py312hca03da5_0 
  nbconvert-core     pkgs/main/osx-arm64::nbconvert-core-7.16.6-py312hca03da5_0 
  nbconvert-pandoc   pkgs/main/osx-arm64::nbconvert-pandoc-7.16.6-py312hca03da5_0 
  nbformat           pkgs/main/osx-arm64::nbformat-5.10.4-py312hca03da5_0 
  nest-asyncio       pkgs/main/osx-arm64::nest-asyncio-1.6.0-py312hca03da5_0 
  notebook           pkgs/main/osx-arm64::notebook-7.4.5-py312hca03da5_0 
  notebook-shim      pkgs/main/osx-arm64::notebook-shim-0.2.4-py312hca03da5_0 
  overrides          pkgs/main/osx-arm64::overrides-7.4.0-py312hca03da5_0 
  packaging          pkgs/main/osx-arm64::packaging-25.0-py312hca03da5_1 
  pandoc             pkgs/main/osx-arm64::pandoc-3.8-hca03da5_0 
  pandocfilters      pkgs/main/osx-arm64::pandocfilters-1.5.1-py312hca03da5_0 
  parso              pkgs/main/osx-arm64::parso-0.8.4-py312hca03da5_0 
  pexpect            pkgs/main/osx-arm64::pexpect-4.9.0-py312hca03da5_0 
  platformdirs       pkgs/main/osx-arm64::platformdirs-4.3.7-py312hca03da5_0 
  prometheus_client  pkgs/main/osx-arm64::prometheus_client-0.21.1-py312hca03da5_0 
  prompt-toolkit     pkgs/main/osx-arm64::prompt-toolkit-3.0.52-py312hca03da5_1 
  prompt_toolkit     pkgs/main/noarch::prompt_toolkit-3.0.52-hd3eb1b0_1 
  psutil             pkgs/main/osx-arm64::psutil-7.0.0-py312haa24f5a_1 
  ptyprocess         pkgs/main/noarch::ptyprocess-0.7.0-pyhd3eb1b0_2 
  pure_eval          pkgs/main/osx-arm64::pure_eval-0.2.3-py312hca03da5_0 
  pycparser          pkgs/main/osx-arm64::pycparser-2.23-py312hca03da5_0 
  pygments           pkgs/main/osx-arm64::pygments-2.19.1-py312hca03da5_0 
  pysocks            pkgs/main/osx-arm64::pysocks-1.7.1-py312hca03da5_1 
  python-dateutil    pkgs/main/osx-arm64::python-dateutil-2.9.0post0-py312hca03da5_2 
  python-fastjsonsc~ pkgs/main/osx-arm64::python-fastjsonschema-2.20.0-py312hca03da5_0 
  python-json-logger pkgs/main/osx-arm64::python-json-logger-3.2.1-py312hca03da5_0 
  pyyaml             pkgs/main/osx-arm64::pyyaml-6.0.2-py312h80987f9_0 
  pyzmq              pkgs/main/osx-arm64::pyzmq-27.1.0-py312h854a7ef_0 
  qtconsole          pkgs/main/osx-arm64::qtconsole-5.7.0-py312hca03da5_0 
  qtpy               pkgs/main/osx-arm64::qtpy-2.4.3-py312hca03da5_0 
  referencing        pkgs/main/osx-arm64::referencing-0.30.2-py312hca03da5_0 
  requests           pkgs/main/osx-arm64::requests-2.32.5-py312hca03da5_0 
  rfc3339-validator  pkgs/main/osx-arm64::rfc3339-validator-0.1.4-py312hca03da5_0 
  rfc3986-validator  pkgs/main/osx-arm64::rfc3986-validator-0.1.1-py312hca03da5_0 
  rpds-py            pkgs/main/osx-arm64::rpds-py-0.22.3-py312h2aea54e_0 
  send2trash         pkgs/main/osx-arm64::send2trash-1.8.2-py312hca03da5_1 
  six                pkgs/main/osx-arm64::six-1.17.0-py312hca03da5_0 
  sniffio            pkgs/main/osx-arm64::sniffio-1.3.0-py312hca03da5_0 
  soupsieve          pkgs/main/osx-arm64::soupsieve-2.5-py312hca03da5_0 
  stack_data         pkgs/main/osx-arm64::stack_data-0.6.3-py312hca03da5_0 
  terminado          pkgs/main/osx-arm64::terminado-0.18.1-py312hca03da5_0 
  tinycss2           pkgs/main/osx-arm64::tinycss2-1.4.0-py312hca03da5_0 
  tornado            pkgs/main/osx-arm64::tornado-6.5.1-py312h80987f9_0 
  traitlets          pkgs/main/osx-arm64::traitlets-5.14.3-py312hca03da5_0 
  typing-extensions  pkgs/main/osx-arm64::typing-extensions-4.15.0-py312hca03da5_0 
  typing_extensions  pkgs/main/osx-arm64::typing_extensions-4.15.0-py312hca03da5_0 
  urllib3            pkgs/main/osx-arm64::urllib3-2.5.0-py312hca03da5_0 
  wcwidth            pkgs/main/osx-arm64::wcwidth-0.2.13-py312hca03da5_0 
  webencodings       pkgs/main/osx-arm64::webencodings-0.5.1-py312hca03da5_2 
  websocket-client   pkgs/main/osx-arm64::websocket-client-1.8.0-py312hca03da5_0 
  widgetsnbextension pkgs/main/osx-arm64::widgetsnbextension-4.0.14-py312hca03da5_0 
  yaml               pkgs/main/osx-arm64::yaml-0.2.5-h1a28f6b_0 
  zeromq             pkgs/main/osx-arm64::zeromq-4.3.5-h2c7f8f0_1 

The following packages will be UPDATED:

  ca-certificates                      2025.7.15-hca03da5_0 --> 2025.11.4-hca03da5_0 
  openssl                                 3.0.17-h4ee41c1_0 --> 3.0.18-h9b4081a_0 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                
Preparing transaction: done                                                     
Verifying transaction: done                                                     
Executing transaction: done                                                     
(py312) liufengxiang@liufengangdeMBP ~ % jupyter --version
Selected Jupyter core packages...
IPython          : 9.5.0
ipykernel        : 6.30.1
ipywidgets       : 8.1.7
jupyter_client   : 8.6.3
jupyter_core     : 5.8.1
jupyter_server   : 2.16.0
jupyterlab       : 4.4.7
nbclient         : 0.10.2
nbconvert        : 7.16.6
nbformat         : 5.10.4
notebook         : 7.4.5
qtconsole        : 5.7.0
traitlets        : 5.14.3
(py312) liufengxiang@liufengangdeMBP ~ % jupyter notebook
[I 2025-11-05 21:10:31.938 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2025-11-05 21:10:31.939 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2025-11-05 21:10:31.941 ServerApp] jupyterlab | extension was successfully linked.
[I 2025-11-05 21:10:31.942 ServerApp] notebook | extension was successfully linked.
[I 2025-11-05 21:10:31.944 ServerApp] Writing Jupyter server cookie secret to /Users/liufengxiang/Library/Jupyter/runtime/jupyter_cookie_secret
[I 2025-11-05 21:10:32.030 ServerApp] notebook_shim | extension was successfully linked.
[I 2025-11-05 21:10:32.047 ServerApp] notebook_shim | extension was successfully loaded.
[I 2025-11-05 21:10:32.048 ServerApp] jupyter_lsp | extension was successfully loaded.
[I 2025-11-05 21:10:32.049 ServerApp] jupyter_server_terminals | extension was successfully loaded.
[I 2025-11-05 21:10:32.050 LabApp] JupyterLab extension loaded from /opt/miniconda3/envs/py312/lib/python3.12/site-packages/jupyterlab
[I 2025-11-05 21:10:32.050 LabApp] JupyterLab application directory is /opt/miniconda3/envs/py312/share/jupyter/lab
[I 2025-11-05 21:10:32.050 LabApp] Extension Manager is 'pypi'.
[I 2025-11-05 21:10:32.100 ServerApp] jupyterlab | extension was successfully loaded.
[I 2025-11-05 21:10:32.102 ServerApp] notebook | extension was successfully loaded.
[I 2025-11-05 21:10:32.103 ServerApp] Serving notebooks from local directory: /Users/liufengxiang
[I 2025-11-05 21:10:32.103 ServerApp] Jupyter Server 2.16.0 is running at:
[I 2025-11-05 21:10:32.103 ServerApp] http://localhost:8888/tree?token=33abd9f33f123b592ec11c207bdcd10fc4d6035777cfb1b0
[I 2025-11-05 21:10:32.103 ServerApp]     http://127.0.0.1:8888/tree?token=33abd9f33f123b592ec11c207bdcd10fc4d6035777cfb1b0
[I 2025-11-05 21:10:32.103 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-11-05 21:10:32.105 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///Users/liufengxiang/Library/Jupyter/runtime/jpserver-6200-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/tree?token=33abd9f33f123b592ec11c207bdcd10fc4d6035777cfb1b0
        http://127.0.0.1:8888/tree?token=33abd9f33f123b592ec11c207bdcd10fc4d6035777cfb1b0
[I 2025-11-05 21:10:32.431 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server

