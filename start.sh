#!/bin/bash

# Create directories
mkdir -p /runpod-volume \
    /root/comfy/ComfyUI/models/loras \
    /root/comfy/ComfyUI/models/diffusion_models \
    /root/comfy/ComfyUI/models/clip \
    /root/comfy/ComfyUI/models/vae

# Validation and download function
validate_or_redownload() {
    file="$1"
    url="$2"
    
    # If file doesn't exist, download it first
    if [ ! -f "$file" ]; then
        echo "Downloading $file"
        wget -O "$file" "$url"
    fi
    
    python3 - <<EOF
from safetensors import safe_open
import sys, os

path = "$file"
try:
    with safe_open(path, framework="pt") as f:
        f.keys()
    print(f"[OK] Valid safetensors file: {path}")
except:
    print(f"[ERROR] Detected corrupted model: {path}")
    os.remove(path)
    print("Re-downloading...")
    import os
    sys.exit(2)  # signal bash to re-download
EOF

    if [ $? -eq 2 ]; then
        wget -O "$file" "$url"
        validate_or_redownload "$file" "$url"
    fi
}

# Download and validate files
validate_or_redownload "/runpod-volume/Wan2.1-Fun-14B-InP-MPS.safetensors" \
  "https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors?download=true"

validate_or_redownload "/runpod-volume/infinitetalk_single.safetensors" \
  "https://huggingface.co/MeiGen-AI/InfiniteTalk/resolve/main/comfyui/infinitetalk_single.safetensors?download=true"

validate_or_redownload "/runpod-volume/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors?download=true"

validate_or_redownload "/runpod-volume/umt5-xxl-enc-bf16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

validate_or_redownload "/runpod-volume/Wan2_1_VAE_bf16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true"

validate_or_redownload "/runpod-volume/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"

# Copy files to ComfyUI directories
cp /runpod-volume/Wan2.1-Fun-14B-InP-MPS.safetensors /root/comfy/ComfyUI/models/loras/
cp /runpod-volume/infinitetalk_single.safetensors /root/comfy/ComfyUI/models/diffusion_models/
cp /runpod-volume/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors /root/comfy/ComfyUI/models/loras/
cp /runpod-volume/umt5-xxl-enc-bf16.safetensors /root/comfy/ComfyUI/models/clip/
cp /runpod-volume/Wan2_1_VAE_bf16.safetensors /root/comfy/ComfyUI/models/vae/
cp /runpod-volume/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors /root/comfy/ComfyUI/models/diffusion_models/

# Validate Lynx folder function
validate_lynx_folder() {
    lynx_path="/runpod-volume/WanVideo_comfy/Lynx"
    
    if [ ! -d "$lynx_path" ]; then
        return 1  # Folder doesn't exist
    fi
    
    # Check if any safetensors files exist in Lynx folder
    if [ ! "$(find "$lynx_path" -name "*.safetensors" | head -1)" ]; then
        return 1  # No safetensors files found
    fi
    
    # Validate each safetensors file in Lynx folder
    find "$lynx_path" -name "*.safetensors" | while read -r file; do
        python3 - <<EOF
from safetensors import safe_open
import sys

path = "$file"
try:
    with safe_open(path, framework="pt") as f:
        f.keys()
    print(f"[OK] Valid safetensors file: {path}")
except:
    print(f"[ERROR] Detected corrupted Lynx model: {path}")
    sys.exit(2)  # signal corruption
EOF
        if [ $? -eq 2 ]; then
            return 1  # Corruption detected
        fi
    done
}

# Download and validate Lynx from HuggingFace Hub
if ! validate_lynx_folder; then
    echo "Lynx folder not found or corrupted, downloading from HuggingFace Hub..."
    rm -rf /runpod-volume/WanVideo_comfy/Lynx
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Kijai/WanVideo_comfy', local_dir='/runpod-volume/WanVideo_comfy', allow_patterns='Lynx/*')"
    
    # Validate again after download
    if ! validate_lynx_folder; then
        echo "[ERROR] Downloaded Lynx files are still corrupted!"
        exit 1
    fi
else
    echo "Lynx folder exists and is valid"
fi

# Copy Lynx model
cp -r /runpod-volume/WanVideo_comfy/Lynx /root/comfy/ComfyUI/models/diffusion_models/

# Start the handler
cd /root/comfy/ComfyUI
python -u rp_handler.py
