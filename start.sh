#!/bin/bash

# Create directories
mkdir -p /runpod-volume \
    /root/comfy/ComfyUI/models/loras \
    /root/comfy/ComfyUI/models/diffusion_models \
    /root/comfy/ComfyUI/models/clip \
    /root/comfy/ComfyUI/models/vae

# Download function
download_if_missing() {
    if [ ! -f "$1" ]; then
        echo "Downloading $1"
        wget -O "$1" "$2"
    else
        echo "Exists: $1"
    fi
}

# Download files
download_if_missing "/runpod-volume/Wan2.1-Fun-14B-InP-MPS.safetensors" \
  "https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors?download=true"

download_if_missing "/runpod-volume/infinitetalk_single.safetensors" \
  "https://huggingface.co/MeiGen-AI/InfiniteTalk/resolve/main/comfyui/infinitetalk_single.safetensors?download=true"

download_if_missing "/runpod-volume/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors?download=true"

download_if_missing "/runpod-volume/umt5-xxl-enc-bf16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

download_if_missing "/runpod-volume/Wan2_1_VAE_bf16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true"

download_if_missing "/runpod-volume/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"

# Copy files to ComfyUI directories
cp /runpod-volume/Wan2.1-Fun-14B-InP-MPS.safetensors /root/comfy/ComfyUI/models/loras/
cp /runpod-volume/infinitetalk_single.safetensors /root/comfy/ComfyUI/models/diffusion_models/
cp /runpod-volume/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors /root/comfy/ComfyUI/models/loras/
cp /runpod-volume/umt5-xxl-enc-bf16.safetensors /root/comfy/ComfyUI/models/clip/
cp /runpod-volume/Wan2_1_VAE_bf16.safetensors /root/comfy/ComfyUI/models/vae/
cp /runpod-volume/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors /root/comfy/ComfyUI/models/diffusion_models/

# Download from HuggingFace Hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Kijai/WanVideo_comfy', local_dir='/runpod-volume/WanVideo_comfy', allow_patterns='Lynx/*')"

# Copy Lynx model
cp -r /runpod-volume/WanVideo_comfy/Lynx /root/comfy/ComfyUI/models/diffusion_models/

# Start the handler
cd /root/comfy/comfyUI
python -u rp_handler.py
