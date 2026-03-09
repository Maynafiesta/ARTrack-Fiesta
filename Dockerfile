# ARG for Blackwell (sm_120) or Jetson (ARM64) support
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files first
COPY requirements.txt .

# Install dependencies including TensorRT requirements
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir onnx onnxsim pycuda cuda-python

# Copy the rest of the application
COPY . .

# Environment paths for evaluation
RUN python3 tracking/create_default_local_file.py --workspace_dir /app --data_dir /app/data --save_dir /app/output

CMD ["bash"]
