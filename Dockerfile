FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set working directory to root
WORKDIR /root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    wget \
    curl \
    build-essential \
    g++ \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
    
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Ensure pip upgraded
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir fastapi[standard]==0.115.4
RUN pip install --no-cache-dir comfy-cli

# Install ComfyUI with fast dependencies and NVIDIA support
RUN comfy --skip-prompt install --fast-deps --nvidia

# Clone ComfyUI custom nodes
RUN git clone https://github.com/kijai/ComfyUI-GIMM-VFI.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-GIMM-VFI && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Easy-Use && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts && \
    git clone https://github.com/sipherxyz/comfyui-art-venture.git /root/comfy/ComfyUI/custom_nodes/comfyui-art-venture && \
    git clone https://github.com/chrisgoringe/cg-use-everywhere.git /root/comfy/ComfyUI/custom_nodes/cg-use-everywhere && \
    git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-mxToolkit && \
    git clone https://github.com/alt-key-project/comfyui-dream-video-batches.git /root/comfy/ComfyUI/custom_nodes/comfyui-dream-video-batches && \
    git clone https://github.com/ciga2011/ComfyUI-MarkItDown.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-MarkItDown && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git clone https://github.com/calcuis/gguf.git /root/comfy/ComfyUI/custom_nodes/gguf && \
    git clone https://github.com/rgthree/rgthree-comfy.git /root/comfy/ComfyUI/custom_nodes/rgthree-comfy

# Install requirements for custom nodes
RUN pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-GIMM-VFI/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-Easy-Use/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/comfyui-art-venture/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/comfyui-dream-video-batches/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-MarkItDown/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials/requirements.txt && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/rgthree-comfy/requirements.txt

# Install additional Python packages
RUN pip install --no-cache-dir huggingface_hub

# Create model directories
RUN mkdir -p /root/comfy/ComfyUI/models/diffusion_models && \
    mkdir -p /root/comfy/ComfyUI/models/loras && \
    mkdir -p /root/comfy/ComfyUI/models/clip && \
    mkdir -p /root/comfy/ComfyUI/models/vae

# Download models and weights


# Clone WanVideoWrapper and install its requirements
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && \
    pip install --no-cache-dir -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# Install final packages
RUN pip install --no-cache-dir sageattention==1.0.6
RUN pip install --no-cache-dir insightface runpod boto3

# Copy rp_handler.py to ComfyUI directory
COPY rp_handler.py /root/comfy/ComfyUI/

# Set the working directory to ComfyUI
WORKDIR /root/comfy/ComfyUI
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
