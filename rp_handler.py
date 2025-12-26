import os
import random
import sys
import time
import uuid
import requests
import subprocess
from typing import Sequence, Mapping, Any, Union
import torch
import boto3
import runpod
import os
import gc
import tts_generator 
import shutil
import psutil
from trim_video import trim_video
from add_silence import add_silence_to_audio
from background_audio import run_pipeline
def purge_vram():
    """Aggressive RAM and VRAM cleanup to prevent OOM"""
    try:
        import gc
        import psutil
        import ctypes
        import sys
        
        # Get initial memory stats
        ram_before = psutil.virtual_memory().used / 1024**3
        
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / 1024**3
            print(f"Before cleanup - RAM: {ram_before:.2f}GB, VRAM: {vram_before:.2f}GB")
            
            # Only clear intermediate VRAM tensors, keep models on GPU
            torch.cuda.empty_cache()
        else:
            print(f"Before cleanup - RAM: {ram_before:.2f}GB")
            
        # AGGRESSIVE RAM CLEANUP
        # Clear all possible Python references
        for _ in range(10):  # Multiple aggressive GC passes
            gc.collect()
            
        # Clear module caches and temporary variables 
        # Clear Python's internal caches
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
            
        # Force memory trim on Linux (returns freed memory to OS)
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            try:
                # Alternative for different systems
                ctypes.CDLL(None).malloc_trim(0)
            except:
                pass
                
        # Final GC passes
        for _ in range(5):
            gc.collect()
            
        # Get final memory stats
        ram_after = psutil.virtual_memory().used / 1024**3
        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated() / 1024**3
            print(f"After cleanup - RAM: {ram_after:.2f}GB (freed {ram_before-ram_after:.2f}GB), VRAM: {vram_after:.2f}GB")
        else:
            print(f"After cleanup - RAM: {ram_after:.2f}GB (freed {ram_before-ram_after:.2f}GB)")
            
        print("Aggressive memory cleanup completed")
        
    except Exception as e:
        print(f"Error during memory cleanup: {e}")


AWS_ACCESS_KEY=os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY")
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

async def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes
    
    sys.path.insert(0, find_path("ComfyUI"))
    import server
    
    loop = asyncio.get_running_loop()
    
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    
    # THIS waits for the async function to finish!
    await init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS
 
async def upscale_workflow(video_file,output_suffix):
    await import_custom_nodes()
    
    # Monitor RAM usage and cleanup if needed
    import psutil
    ram_usage = psutil.virtual_memory().percent
    if ram_usage > 70:
        print(f"Warning: High RAM usage ({ram_usage:.1f}%), performing cleanup")
        purge_vram()
    
    # Configure PyTorch for optimal VRAM usage
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    # Set memory allocation strategy to prefer VRAM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)  # Use 90% of VRAM, leave 10% buffer
        torch.cuda.empty_cache()  # Start with clean VRAM
    
    with torch.inference_mode():
        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_1 = vhs_loadvideo.load_video(
            video=video_file,
            force_rate=0,
            custom_width=0,
            custom_height=0,
            frame_load_cap=0,
            skip_first_frames=0,
            select_every_nth=1,
            format="AnimateDiff",
            unique_id=14884766115541647826,
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_4 = upscalemodelloader.EXECUTE_NORMALIZED(
            model_name="RealESRGAN_x2.pth"
        )

        vhs_videoinfosource = NODE_CLASS_MAPPINGS["VHS_VideoInfoSource"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            vhs_videoinfosource_3 = vhs_videoinfosource.get_video_info(
                video_info=get_value_at_index(vhs_loadvideo_1, 3)
            )

            imageupscalewithmodel_7 = imageupscalewithmodel.EXECUTE_NORMALIZED(
                upscale_model=get_value_at_index(upscalemodelloader_4, 0),
                image=get_value_at_index(vhs_loadvideo_1, 0),
            )

            imagescaleby_6 = imagescaleby.upscale(
                upscale_method="bicubic",
                scale_by=1,
                image=get_value_at_index(imageupscalewithmodel_7, 0),
            )

            vhs_videocombine_2 = vhs_videocombine.combine_video(
                frame_rate=get_value_at_index(vhs_videoinfosource_3, 0),
                loop_count=0,
                filename_prefix=f"output_{output_suffix}",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(imagescaleby_6, 0),
                unique_id=4842594312946713770,
            )
async def workflow(prompt:str,prompt_motion:str,audio_file,image_file, output_suffix):
    await import_custom_nodes()
    
    # Monitor RAM usage and cleanup if needed
    import psutil
    ram_usage = psutil.virtual_memory().percent
    if ram_usage > 70:
        print(f"Warning: High RAM usage ({ram_usage:.1f}%), performing cleanup")
        purge_vram()
    
    # Configure PyTorch for optimal VRAM usage
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    # Set memory allocation strategy to prefer VRAM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)  # Use 90% of VRAM, leave 10% buffer
        torch.cuda.empty_cache()  # Start with clean VRAM
    
    with torch.inference_mode():
        loadlynxresampler = NODE_CLASS_MAPPINGS["LoadLynxResampler"]()
        loadlynxresampler_16 = loadlynxresampler.loadmodel(
            model_name="Lynx/lynx_lite_resampler_fp32.safetensors", precision="fp16"
        )

        wanvideotextencodecached = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
        wanvideotextencodecached_26 = wanvideotextencodecached.process(
            model_name="umt5-xxl-enc-bf16.safetensors",
            precision="bf16",
            positive_prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            quantization="disabled",
            use_disk_cache=True,
            device="gpu",
        )

        wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        wanvideoblockswap_29 = wanvideoblockswap.setargs(
            blocks_to_swap=5,  # Reduced from 20 to keep more blocks in VRAM
            offload_img_emb=False,  # Keep image embeddings in VRAM
            offload_txt_emb=False,  # Keep text embeddings in VRAM
            use_non_blocking=True,
            vace_blocks_to_swap=0,  # Don't swap VAE blocks
            prefetch_blocks=3,  # Increased prefetch for better VRAM utilization
            block_swap_debug=False,
        )

        wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        wanvideovaeloader_31 = wanvideovaeloader.loadmodel(
            model_name="Wan2_1_VAE_bf16.safetensors", precision="bf16"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_35 = loadimage.load_image(
            image=image_file
        )

        wanvideotextencodecached_51 = wanvideotextencodecached.process(
            model_name="umt5-xxl-enc-bf16.safetensors",
            precision="bf16",
            positive_prompt="image of a face",
            negative_prompt="",
            quantization="disabled",
            use_disk_cache=True,
            device="gpu",
        )

        lynxinsightfacecrop = NODE_CLASS_MAPPINGS["LynxInsightFaceCrop"]()
        lynxinsightfacecrop_53 = lynxinsightfacecrop.encode(
            image=get_value_at_index(loadimage_35, 0)
        )

        lynxencodefaceip = NODE_CLASS_MAPPINGS["LynxEncodeFaceIP"]()
        lynxencodefaceip_52 = lynxencodefaceip.encode(
            resampler=get_value_at_index(loadlynxresampler_16, 0),
            ip_image=get_value_at_index(lynxinsightfacecrop_53, 0),
        )

        wanvideoextramodelselect = NODE_CLASS_MAPPINGS["WanVideoExtraModelSelect"]()
        wanvideoextramodelselect_69 = wanvideoextramodelselect.getmodelpath(
            extra_model="Lynx/Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors"
        )

        wanvideoloraselectmulti = NODE_CLASS_MAPPINGS["WanVideoLoraSelectMulti"]()
        wanvideoloraselectmulti_74 = wanvideoloraselectmulti.getlorapath(
            lora_0="lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors",
            strength_0=1,
            lora_1="Wan2.1-Fun-14B-InP-MPS.safetensors",
            strength_1=0.5,
            lora_2="none",
            strength_2=1,
            lora_3="none",
            strength_3=1,
            lora_4="none",
            strength_4=1,
            low_mem_load=False,
            merge_loras=False,
        )

        vhs_loadaudioupload = NODE_CLASS_MAPPINGS["VHS_LoadAudioUpload"]()
        vhs_loadaudioupload_93 = vhs_loadaudioupload.load_audio(
            audio=audio_file, start_time=0, duration=0
        )

        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        intconstant_95 = intconstant.get_value(value=24)

        downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadWav2VecModel"
        ]()
        downloadandloadwav2vecmodel_96 = downloadandloadwav2vecmodel.loadmodel(
            model="TencentGameMate/chinese-wav2vec2-base",
            base_precision="fp16",
            load_device="main_device",  # Force GPU to avoid RAM usage
        )

        multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
        multitalkmodelloader_100 = multitalkmodelloader.loadmodel(
            model="infinitetalk_single.safetensors"
        )
        
        # Clean RAM after loading heavy models
        import gc
        for _ in range(3):
            gc.collect()

        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_103 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        
        

        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        wanvideosetloras = NODE_CLASS_MAPPINGS["WanVideoSetLoRAs"]()
        wanvideosetblockswap = NODE_CLASS_MAPPINGS["WanVideoSetBlockSwap"]()
        wanvideoemptyembeds = NODE_CLASS_MAPPINGS["WanVideoEmptyEmbeds"]()
        wanvideoaddlynxembeds = NODE_CLASS_MAPPINGS["WanVideoAddLynxEmbeds"]()
        multitalkwav2vecembeds = NODE_CLASS_MAPPINGS["MultiTalkWav2VecEmbeds"]()
        wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        imageselector = NODE_CLASS_MAPPINGS["ImageSelector"]()
        wanimagetovideo = NODE_CLASS_MAPPINGS["WanImageToVideo"]()
        pathchsageattentionkj = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imagebatchextendwithoverlap = NODE_CLASS_MAPPINGS[
            "ImageBatchExtendWithOverlap"
        ]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            wanvideoextramodelselect_13 = wanvideoextramodelselect.getmodelpath(
                extra_model="Lynx/Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors",
                prev_model=get_value_at_index(wanvideoextramodelselect_69, 0),
            )

            wanvideomodelloader_12 = wanvideomodelloader.loadmodel(
                model="Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                base_precision="fp16_fast",
                quantization="disabled",
                load_device="main_device",  # Changed from offload_device to keep models on GPU
                attention_mode="sageattn",
                rms_norm_function="default",
                extra_model=get_value_at_index(wanvideoextramodelselect_13, 0),
                multitalk_model=get_value_at_index(multitalkmodelloader_100, 0),
            )

            wanvideosetloras_60 = wanvideosetloras.setlora(
                model=get_value_at_index(wanvideomodelloader_12, 0),
                lora=get_value_at_index(wanvideoloraselectmulti_74, 0),
            )

            wanvideosetblockswap_28 = wanvideosetblockswap.loadmodel(
                model=get_value_at_index(wanvideosetloras_60, 0),
                block_swap_args=get_value_at_index(wanvideoblockswap_29, 0),
            )

            wanvideoemptyembeds_24 = wanvideoemptyembeds.process(
                width=352, height=624, num_frames=int(get_value_at_index(intconstant_95, 0)*get_value_at_index(vhs_loadaudioupload_93, 1))
            )

            wanvideoaddlynxembeds_55 = wanvideoaddlynxembeds.add(
                ip_scale=0.7,
                ref_scale=0.6,
                lynx_cfg_scale=2,
                start_percent=0,
                end_percent=1,
                embeds=get_value_at_index(wanvideoemptyembeds_24, 0),
                vae=get_value_at_index(wanvideovaeloader_31, 0),
                lynx_ip_embeds=get_value_at_index(lynxencodefaceip_52, 0),
                ref_image=get_value_at_index(lynxinsightfacecrop_53, 1),
                ref_text_embed=get_value_at_index(wanvideotextencodecached_51, 0),
            )

            multitalkwav2vecembeds_99 = multitalkwav2vecembeds.process(
                normalize_loudness=True,
                num_frames=int(get_value_at_index(intconstant_95, 0)*get_value_at_index(vhs_loadaudioupload_93, 1)),
                fps=24.9,
                audio_scale=1,
                audio_cfg_scale=2,
                multi_audio_type="para",
                wav2vec_model=get_value_at_index(downloadandloadwav2vecmodel_96, 0),
                audio_1=get_value_at_index(vhs_loadaudioupload_93, 0),
            )

            wanvideosampler_22 = wanvideosampler.process(
                steps=5,
                cfg=1,
                shift=8,
                seed=random.randint(1, 2**64),
                force_offload=False,
                scheduler="lcm",
                riflex_freq_index=0,
                denoise_strength=1,
                batched_cfg=False,
                rope_function="comfy",
                start_step=0,
                end_step=-1,
                add_noise_to_samples=False,
                model=get_value_at_index(wanvideosetblockswap_28, 0),
                image_embeds=get_value_at_index(wanvideoaddlynxembeds_55, 0),
                text_embeds=get_value_at_index(wanvideotextencodecached_26, 0),
                multitalk_embeds=get_value_at_index(multitalkwav2vecembeds_99, 0),
            )

            wanvideodecode_32 = wanvideodecode.decode(
                enable_vae_tiling=True,  # Enable tiling to save VRAM during decode
                tile_x=512,  # Larger tiles for better VRAM efficiency
                tile_y=512,  # Larger tiles for better VRAM efficiency
                tile_stride_x=256,  # Optimized stride
                tile_stride_y=256,  # Optimized stride
                normalization="default",
                vae=get_value_at_index(wanvideovaeloader_31, 0),
                samples=get_value_at_index(wanvideosampler_22, 0),
            )
            
            # Monitor and cleanup RAM if needed
            vram_usage_start = torch.cuda.memory_allocated() / 1024**3
            del (
                wanvideoextramodelselect_13,
                wanvideomodelloader_12,
                wanvideosetloras_60,
                wanvideosetblockswap_28,
                wanvideoemptyembeds_24,
                wanvideoaddlynxembeds_55,
                multitalkwav2vecembeds_99,
                wanvideosampler_22,
            )
            purge_vram() 
            vram_usage_end = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM usage before cleanup: {vram_usage_start:.2f}GB, after cleanup: {vram_usage_end:.2f}GB")
            print("Selecting image...")
            imageselector_101 = imageselector.run(
                selected_indexes="-1", images=get_value_at_index(wanvideodecode_32, 0)
            )
            print("Loading umt5_xxl_fp8_e4m3fn_scaled.safetensors")
            cliploader_116 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",  # Use GPU instead of CPU
            )
            print("Encoding negative prompt text...")
            cliptextencode_102 = cliptextencode.encode(
                text="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                clip=get_value_at_index(cliploader_116, 0),
            )
            print("Encoding motion prompt text...")
            cliptextencode_121 = cliptextencode.encode(
                text=prompt_motion,
                clip=get_value_at_index(cliploader_116, 0),
            )
            del cliploader_116
            purge_vram()
            print("Generating image to video latent...")
            wanimagetovideo_105 = wanimagetovideo.EXECUTE_NORMALIZED(
                width=352,
                height=624,
                length=121,
                batch_size=1,
                positive=get_value_at_index(cliptextencode_121, 0),
                negative=get_value_at_index(cliptextencode_102, 0),
                vae=get_value_at_index(vaeloader_103, 0),
                start_image=get_value_at_index(imageselector_101, 0),
            )
            print("Loading 2_i2v_high_noise_14B_fp8_scaled.safetensors")
            unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            unetloader_114 = unetloader.load_unet(
                unet_name="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                weight_dtype="fp8_e4m3fn",
            )
            
            print("Applying LoRA and Sage Attention patches...")
            print("loading Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")
            loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            loraloadermodelonly_117 = loraloadermodelonly.load_lora_model_only(
                lora_name="Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                strength_model=1,
                model=get_value_at_index(unetloader_114, 0),
            )
            pathchsageattentionkj_112 = pathchsageattentionkj.patch(
                sage_attention="auto",
                allow_compile=False,
                model=get_value_at_index(loraloadermodelonly_117, 0),
            )

            modelsamplingsd3_106 = modelsamplingsd3.patch(
                shift=8.000000000000002,
                model=get_value_at_index(pathchsageattentionkj_112, 0),
            )
            
            print("Starting advanced k-sampling...")
            ksampleradvanced_113 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=6,
                cfg=1,
                sampler_name="lcm",
                scheduler="simple",
                start_at_step=0,
                end_at_step=3,
                return_with_leftover_noise="enable",
                model=get_value_at_index(modelsamplingsd3_106, 0),
                positive=get_value_at_index(wanimagetovideo_105, 0),
                negative=get_value_at_index(wanimagetovideo_105, 1),
                latent_image=get_value_at_index(wanimagetovideo_105, 2),
            )
            del unetloader_114
            del loraloadermodelonly_117
            purge_vram() 
            print("Loading 2_i2v_low_noise_14B_fp8_scaled.safetensors")
            unetloader_115 = unetloader.load_unet(
                unet_name="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                weight_dtype="fp8_e4m3fn",
            )
            print("Loading Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")
            loraloadermodelonly_118 = loraloadermodelonly.load_lora_model_only(
                lora_name="Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
                strength_model=1,
                model=get_value_at_index(unetloader_115, 0),
            )    
            pathchsageattentionkj_108 = pathchsageattentionkj.patch(
                sage_attention="auto",
                allow_compile=False,
                model=get_value_at_index(loraloadermodelonly_118, 0),
            )

            modelsamplingsd3_109 = modelsamplingsd3.patch(
                shift=8, model=get_value_at_index(pathchsageattentionkj_108, 0)
            )
            ksampleradvanced_110 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=6,
                cfg=1, 
                sampler_name="lcm",
                scheduler="simple",
                start_at_step=3,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(modelsamplingsd3_109, 0),
                positive=get_value_at_index(wanimagetovideo_105, 0),
                negative=get_value_at_index(wanimagetovideo_105, 1),
                latent_image=get_value_at_index(ksampleradvanced_113, 0),
            )

            vaedecode_111 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_110, 0),
                vae=get_value_at_index(vaeloader_103, 0),
            ) 
            imagebatchextendwithoverlap_126 = (
                imagebatchextendwithoverlap.imagesfrombatch(
                    overlap=1,
                    overlap_side="source",
                    overlap_mode="linear_blend",
                    source_images=get_value_at_index(wanvideodecode_32, 0),
                    new_images=get_value_at_index(vaedecode_111, 0),
                )
            )

            vhs_videocombine_120 = vhs_videocombine.combine_video(
                frame_rate=24,
                loop_count=0,
                filename_prefix=f"output_{output_suffix}",
                format="video/h265-mp4",
                pix_fmt="yuv420p",
                crf=22,
                save_metadata=True,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(imagebatchextendwithoverlap_126, 2),
                audio=get_value_at_index(vhs_loadaudioupload_93, 0),
                unique_id=3956264520723635728,
            )

async def handler(input):
    # Monitor RAM at start
    ram_usage = psutil.virtual_memory().percent
    print(f"Starting handler - RAM usage: {ram_usage:.1f}%")
    
    if ram_usage > 75:
        print("High RAM usage detected, performing emergency cleanup")
        purge_vram()

    prompt = input["input"].get("prompt")
    prompt_motion = input["input"].get("prompt_motion")
    image_url = input["input"].get("image_url")
    audio_dialog = input["input"].get("dialog")
    gender = input["input"].get("gender", "female")
    if gender == "male":
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091"
    else:
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9"
    user_id=uuid.uuid4().hex[:8]
    # Ensure inputs directory exists before any file operations
    inputs_dir = "/root/comfy/ComfyUI/input/"
    os.makedirs(inputs_dir, exist_ok=True)
    audio_url = input["input"].get("audio_url")
    if audio_url is not None:
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        audio_filename = f"{user_id}_audio.wav"
        audio_path = os.path.join(inputs_dir, audio_filename)
        with open(audio_path, 'wb') as f:
            f.write(audio_response.content)
    else:
        audio_path = tts_generator.generate_audio_from_transcript(audio_dialog, output_filename=f"{user_id}_dialog_audio", voice_id=voice_id)
    add_silence_to_audio(audio_path,audio_path)
    
    # Download image
    image_response = requests.get(image_url)
    image_response.raise_for_status()
    
    # Get file extension from URL or default to .jpg for image
    image_ext = os.path.splitext(image_url.split('?')[0])[1] or '.jpg'
    image_filename = f"{uuid.uuid4().hex}{image_ext}"
    image_path = os.path.join(inputs_dir, image_filename)
    
    with open(image_path, 'wb') as f:
        f.write(image_response.content)
    
    # Copy the generated TTS audio to the inputs directory
    tts_audio_filename = os.path.basename(audio_path)
    inputs_audio_path = os.path.join(inputs_dir, tts_audio_filename)
    shutil.copy2(audio_path, inputs_audio_path)
    audio_path = inputs_audio_path
    
    random_suffix = uuid.uuid4().hex[:6]
    # Run the workflow with the downloaded files (pass filenames, not full paths)
    await workflow(prompt, prompt_motion, tts_audio_filename, image_filename, random_suffix)
    
    # Find the output video file
    outputs_dir = "/root/comfy/ComfyUI/output/"
    output_filename = None

    if os.path.exists(outputs_dir):
        for filename in os.listdir(outputs_dir):
            if filename.startswith(f"output_{random_suffix}") and filename.endswith(".mp4"):
                output_filename = filename
                break

    if output_filename:
        output_path = os.path.join(outputs_dir, output_filename)
    # Copy the output video file to the inputs directory
    else:
        output_path = None
    shutil.copy2(output_path, os.path.join(inputs_dir,output_filename))
    random_suffix = uuid.uuid4().hex[:6]+"_upscaled"
    # Uncomment the following lines to enable upscaling
    # await upscale_workflow(output_filename, random_suffix)
    # # Find the upscaled output video file
    # output_filename = None
    # if os.path.exists(outputs_dir):
    #     for filename in os.listdir(outputs_dir):
    #         if filename.startswith(f"output_{random_suffix}") and filename.endswith(".mp4"):
    #             output_filename = filename
    #             break
    # if output_filename:
    #     output_path = os.path.join(outputs_dir, output_filename)
        
    final_video_path = None
    # Process video with ffmpeg to ensure audio is properly combined
    if output_path and os.path.exists(output_path):
        final_video_filename = f"final_{random_suffix}.mp4"
        final_video_path = os.path.join(outputs_dir, final_video_filename)
        
        try:
            # Use ffmpeg to combine video and audio with proper encoding
            # Audio will play at start, then silence for rest of video duration
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # -y to overwrite output files
                '-i', output_path,  # Input video
                '-i', audio_path,   # Input audio
                '-c:v', 'copy',     # Copy video stream without re-encoding to preserve original
                '-c:a', 'aac',      # Audio codec
                '-b:a', '128k',     # Audio bitrate
                '-filter_complex', '[1:a]apad=whole_dur=1800[padded_audio]',  # Pad audio to 30 minutes max (1800 seconds)
                '-map', '0:v',      # Map video from first input
                '-map', '[padded_audio]',  # Map padded audio
                '-shortest',        # End when shortest input ends (video determines final duration)
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                final_video_path
            ]
            
            print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True, timeout=60)  # 60 second timeout
            print(f"FFmpeg completed successfully")
            
        except subprocess.TimeoutExpired:
            print("FFmpeg timed out after 60 seconds")
            # Kill any ffmpeg processes and use original video
            subprocess.run(['pkill', '-f', 'ffmpeg'], capture_output=True)
            final_video_path = output_path
            final_video_filename = output_filename
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            # If ffmpeg fails, use the original video
            final_video_path = output_path
            final_video_filename = output_filename
        except Exception as e:
            print(f"Error processing video with ffmpeg: {e}")
            # If ffmpeg fails, use the original video
            final_video_path = output_path
            final_video_filename = output_filename 
     
    final_final_video_path = os.path.join(outputs_dir, f"full_video_{random_suffix}.mp4")
    run_pipeline(final_video_path,output_path=final_final_video_path)
    trim_video(final_final_video_path,final_final_video_path)
    # Upload output video to S3 and get URL
    if final_final_video_path and os.path.exists(final_final_video_path):
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name='eu-north-1'  
        )
         
        bucket_name = 'fritz-comfyui'  
        s3_key = f"videos/{final_video_filename}"
        
        try:
            # Upload file to S3
            print(f"Uploading {final_final_video_path} to S3 as {s3_key}")
            s3_client.upload_file(final_final_video_path, bucket_name, s3_key)
            
            # Generate S3 URL
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            
            output_s3_url = s3_url
            print(f"Successfully uploaded to S3: {s3_url}")
            
            # Clean up local files
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if final_video_path != output_path and os.path.exists(output_path):
                    os.remove(output_path)  # Remove original if we created a new one
                if os.path.exists(final_video_path):
                    os.remove(final_video_path)  # Remove final after upload
            except Exception as e:
                print(f"Error cleaning up files: {e}")
                
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            output_s3_url = None
    else:
        print("No output video file found")
        output_s3_url = None
    purge_vram()
    return { 
        "message": "Success! Download the video from the provided URL." if output_s3_url else "Failed to generate or upload video.",
        "video_url": output_s3_url
    }
 
runpod.serverless.start({"handler": handler}) 
