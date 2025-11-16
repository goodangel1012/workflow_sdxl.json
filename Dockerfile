# Use Python 3.12 Debian slim base image
FROM python:3.12-slim

# Set working directory
WORKDIR /root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

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
RUN wget -O /root/comfy/ComfyUI/models/loras/Wan2.1-Fun-14B-InP-MPS.safetensors \
    "https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors?download=true" && \
    wget -O /root/comfy/ComfyUI/models/diffusion_models/infinitetalk_single.safetensors \
    "https://huggingface.co/MeiGen-AI/InfiniteTalk/resolve/main/comfyui/infinitetalk_single.safetensors?download=true" && \
    wget -O /root/comfy/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors?download=true" && \
    wget -O /root/comfy/ComfyUI/models/clip/umt5-xxl-enc-bf16.safetensors \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" && \
    wget -O /root/comfy/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true" && \
    wget -O /root/comfy/ComfyUI/models/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"

# Download using huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Kijai/WanVideo_comfy', local_dir='/root/comfy/ComfyUI/models/diffusion_models', allow_patterns='Lynx/*')"

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

CMD ["python", "-u", "rp_handler.py"]
