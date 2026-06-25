FROM python:3.10-slim

# 安装所有系统依赖（彻底解决 libGL 问题）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先强制安装 opencv-python-headless
RUN pip install opencv-python-headless==4.11.0.86

# 复制 requirements.txt（里面不要包含 opencv 相关行）
COPY requirements.txt .

# 强制用 --no-deps 安装 ultralytics 和 streamlit-webrtc
RUN pip install --no-cache-dir --no-deps ultralytics==8.4.40
RUN pip install --no-cache-dir --no-deps streamlit-webrtc==0.72.2

# 安装其他所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main3.py", "--server.port=8501", "--server.address=0.0.0.0"]
