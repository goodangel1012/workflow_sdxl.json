#!/bin/bash

pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python3 -c "import torch; print('Torch:', torch.__version__); print('Compiled CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); import sys; sys.stdout.flush(); exec('try:\\n    print(\\'Device count:\\', torch.cuda.device_count())\\nexcept Exception as e:\\n    print(\\'Device count error:\\', e)')"

# Create directories
mkdir -p /runpod-volume \
    /root/comfy/ComfyUI/models/loras \
    /root/comfy/ComfyUI/models/diffusion_models \
    /root/comfy/ComfyUI/models/clip \
    /root/comfy/ComfyUI/models/clip_vision \
    /root/comfy/ComfyUI/models/vae \
    /root/comfy/ComfyUI/custom_nodes

# Install custom nodes
echo "Installing custom nodes..."

# ComfyUI-to-Python-Extension
if [ ! -d "/root/comfy/ComfyUI/custom_nodes/ComfyUI-to-Python-Extension" ]; then
    echo "Installing ComfyUI-to-Python-Extension..."
    git clone https://github.com/pydn/ComfyUI-to-Python-Extension.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-to-Python-Extension
fi

# ComfyUI-PixtralLlamaMolmoVision
if [ ! -d "/root/comfy/ComfyUI/custom_nodes/ComfyUI-PixtralLlamaMolmoVision" ]; then
    echo "Installing ComfyUI-PixtralLlamaMolmoVision..."
    git clone https://github.com/SeanScripts/ComfyUI-PixtralLlamaMolmoVision.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-PixtralLlamaMolmoVision
    if [ -f "/root/comfy/ComfyUI/custom_nodes/ComfyUI-PixtralLlamaMolmoVision/requirements.txt" ]; then
        pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-PixtralLlamaMolmoVision/requirements.txt
    fi
fi

# ComfyUI-Image-Selector
if [ ! -d "/root/comfy/ComfyUI/custom_nodes/ComfyUI-Image-Selector" ]; then
    echo "Installing ComfyUI-Image-Selector..."
    git clone https://github.com/SLAPaper/ComfyUI-Image-Selector.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-Image-Selector
fi

# Validation and download function
validate_or_redownload() {
    file="$1"
    url="$2"
    
    # Ensure directory exists before downloading
    mkdir -p "$(dirname "$file")"
    
    # If file doesn't exist, download it first
    if [ ! -f "$file" ]; then
        echo "Downloading $file"
        wget -O "$file" "$url"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Download failed for $file"
            return 1
        fi
    fi
     
    python3 - <<EOF
from safetensors import safe_open
import sys, os

path = "$file"
try:
    with safe_open(path, framework="pt") as f:
        f.keys()
    print(f"[OK] Valid safetensors file: {path}")
except Exception as e:
    print(f"[ERROR] Detected corrupted model: {path} - {e}")
    if os.path.exists(path):
        os.remove(path)
    print("Re-downloading...")
    sys.exit(2)  # signal bash to re-download
EOF

    if [ $? -eq 2 ]; then
        echo "Re-downloading $file due to corruption..."
        # Remove any existing file before re-downloading
        rm -f "$file"
        wget -O "$file" "$url"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Re-download failed for $file"
            return 1
        fi
        validate_or_redownload "$file" "$url"
    fi
}

# Download and validate files
validate_or_redownload "/runpod-volume/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors"
 
validate_or_redownload "/runpod-volume/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

validate_or_redownload "/runpod-volume/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

validate_or_redownload "/runpod-volume/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors" \
  "https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/resolve/main/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"

validate_or_redownload "/runpod-volume/wan_2.1_vae.safetensors" \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

validate_or_redownload "/runpod-volume/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "https://huggingface.co/chatpig/encoder/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

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
echo "Copying models to ComfyUI directories..."

# Copy new models
cp /runpod-volume/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors /root/comfy/ComfyUI/models/clip/
cp /runpod-volume/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors /root/comfy/ComfyUI/models/clip_vision/ 
cp /runpod-volume/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors /root/comfy/ComfyUI/models/diffusion_models/
cp /runpod-volume/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors /root/comfy/ComfyUI/models/diffusion_models/
cp /runpod-volume/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors /root/comfy/ComfyUI/models/loras/
cp /runpod-volume/wan_2.1_vae.safetensors /root/comfy/ComfyUI/models/vae/
cp /runpod-volume/umt5_xxl_fp8_e4m3fn_scaled.safetensors /root/comfy/ComfyUI/models/clip/

# Copy existing models
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
    
    # Define required Lynx model files
    required_lynx_files=(
        "Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors"
        "lynx_lite_resampler_fp32.safetensors"
        "Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors"
        # Add other required Lynx model files here as needed
    )
    
    # Check if all required files exist
    for file in "${required_lynx_files[@]}"; do
        if [ ! -f "$lynx_path/$file" ]; then
            echo "[ERROR] Required Lynx model file not found: $file"
            return 1
        fi
    done
    
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
