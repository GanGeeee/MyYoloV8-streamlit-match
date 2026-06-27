FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制 requirements.txt
COPY requirements.txt .

# ========== 方案：直接用 requirements.txt 安装所有依赖 ==========
# 注意：requirements.txt 中已经包含了 opencv-python-headless 和 ultralytics
RUN pip install --no-cache-dir -r requirements.txt

# ========== 如果上面的命令失败，用备用方案 ==========
# 先安装基础包
RUN pip install --no-cache-dir opencv-python-headless==4.11.0.86 || echo "OpenCV install failed"
RUN pip install --no-cache-dir ultralytics==8.4.40 || echo "Ultralytics install failed"
RUN pip install --no-cache-dir streamlit-webrtc==0.72.2 || echo "streamlit-webrtc install failed"

# 验证安装
RUN python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" || echo "❌ OpenCV not installed"
RUN python -c "from ultralytics import YOLO; print('✅ Ultralytics imported')" || echo "❌ Ultralytics not installed"

# 复制所有代码
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main3.py", "--server.port=8501", "--server.address=0.0.0.0"]
