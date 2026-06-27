FROM python:3.10-slim

# 安装所有系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制 requirements.txt
# COPY requirements.txt .

# ========== 关键：分步安装，确保 opencv-headless 不被覆盖 ==========

# 第一步：单独安装 opencv-python-headless（不带任何依赖）
RUN pip install opencv-python-headless==4.11.0.86


# 第三步：安装 ultralytics（不带依赖，手动控制）
RUN pip install ultralytics

# 第四步：安装 streamlit-webrtc（不带依赖，手动控制）
RUN pip install streamlit-webrtc==0.72.2

# 第五步：安装其他所有依赖（注意：不要覆盖 opencv）
RUN pip install \
    absl-py==2.4.0 \
    aioice==0.10.2 \
    aiortc==1.14.0 \
    altair==6.0.0 \
    antlr4-python3-runtime==4.9.3 \
    asttokens==3.0.1 \
    attrs==26.1.0 \
    av==16.1.0 \
    blinker==1.9.0 \
    cachetools==7.0.5 \
    certifi==2026.2.25 \
    cffi==2.0.0 \
    charset-normalizer==3.4.7 \
    click==8.3.2 \
    colorama==0.4.6 \
    contourpy==1.3.2 \
    cryptography==49.0.0 \
    cycler==0.12.1 \
    decorator==5.2.1 \
    dnspython==2.8.0 \
    exceptiongroup==1.3.1 \
    executing==2.2.1 \
    filelock==3.25.2 \
    fonttools==4.62.1 \
    fsspec==2026.2.0 \
    gitdb==4.0.12 \
    GitPython==3.1.46 \
    google-crc32c==1.8.0 \
    grpcio==1.80.0 \
    hydra-core==1.3.2 \
    idna==3.11 \
    ifaddr==0.2.0 \
    ipython==8.39.0 \
    jedi==0.19.2 \
    Jinja2==3.1.6 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    kiwisolver==1.5.0 \
    Markdown==3.10.2 \
    MarkupSafe==3.0.3 \
    matplotlib==3.10.8 \
    matplotlib-inline==0.2.1 \
    mpmath==1.3.0 \
    mss==10.1.0 \
    narwhals==2.19.0 \
    networkx==3.4.2 \
    numpy==1.23.5 \
    omegaconf==2.3.0 \
    packaging==26.0 \
    pandas==2.3.3 \
    parso==0.8.6 \
    pexpect==4.9.0 \
    Pillow==9.5.0 \
    polars==1.40.0 \
    polars-runtime-32==1.40.0 \
    prompt_toolkit==3.0.52 \
    protobuf==7.34.1 \
    psutil==7.2.2 \
    ptyprocess==0.7.0 \
    pure_eval==0.2.3 \
    pyarrow==23.0.1 \
    pycparser==3.0 \
    pydeck==0.9.2 \
    pyee==13.0.1 \
    Pygments==2.20.0 \
    pylibsrtp==1.0.0 \
    pyOpenSSL==26.3.0 \
    pyparsing==3.3.2 \
    python-dateutil==2.9.0.post0 \
    pytz==2026.1.post1 \
    PyYAML==6.0.3 \
    referencing==0.37.0 \
    requests==2.33.1 \
    rpds-py==0.30.0 \
    scipy==1.15.3 \
    seaborn==0.13.2 \
    six==1.17.0 \
    smmap==5.0.3 \
    stack-data==0.6.3 \
    streamlit==1.56.0 \
    sympy==1.14.0 \
    tenacity==9.1.4 \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    thop==0.1.1.post2209072238 \
    toml==0.10.2 \
    torch==2.4.0 \
    torchvision==0.19.0 \
    tornado==6.5.5 \
    tqdm==4.67.3 \
    traitlets==5.14.3 \
    typing_extensions==4.15.0 \
    tzdata==2026.1 \
    ultralytics==8.4.40 \
    ultralytics-thop==2.0.18 \
    urllib3==2.6.3 \
    watchdog==6.0.0 \
    wcwidth==0.6.0 \
    Werkzeug==3.1.8

# 第六步：再次验证 cv2 是否仍然可用
RUN python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# 复制所有代码
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main/main3.py", "--server.port=8501", "--server.address=0.0.0.0"]
