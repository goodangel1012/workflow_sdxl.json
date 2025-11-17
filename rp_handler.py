import os
import random
import sys
import uuid
import requests
from typing import Sequence, Mapping, Any, Union
import torch
import boto3
import runpod
import os

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


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes

    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS


def workflow(prompt:str,audio_file,image_file, output_suffix):
    import_custom_nodes()
    with torch.inference_mode():
        loadlynxresampler = NODE_CLASS_MAPPINGS["LoadLynxResampler"]()
        loadlynxresampler_16 = loadlynxresampler.loadmodel(
            model_name="Lynx/lynx_lite_resampler_fp32.safetensors", precision="fp16"
        )

        wanvideoemptyembeds = NODE_CLASS_MAPPINGS["WanVideoEmptyEmbeds"]()
        

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
            blocks_to_swap=20,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
            prefetch_blocks=1,
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
            audio=audio_file,
            start_time=0,
            duration=0,
        )

        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        intconstant_95 = intconstant.get_value(value=24)
        
        mathexpressionpysssss_98: 94 = mathexpressionpysssss.evaluate(
            expression="a*b",
            a=get_value_at_index(intconstant_95, 0),
            b=get_value_at_index(vhs_loadaudioupload_93, 1),
        )
        downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadWav2VecModel"
        ]()
        downloadandloadwav2vecmodel_96 = downloadandloadwav2vecmodel.loadmodel(
            model="TencentGameMate/chinese-wav2vec2-base",
            base_precision="fp16",
            load_device="main_device",
        )

        multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
        multitalkmodelloader_100 = multitalkmodelloader.loadmodel(
            model="infinitetalk_single.safetensors"
        )

        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        wanvideosetloras = NODE_CLASS_MAPPINGS["WanVideoSetLoRAs"]()
        wanvideosetblockswap = NODE_CLASS_MAPPINGS["WanVideoSetBlockSwap"]()
        wanvideoaddlynxembeds = NODE_CLASS_MAPPINGS["WanVideoAddLynxEmbeds"]()
        multitalkwav2vecembeds = NODE_CLASS_MAPPINGS["MultiTalkWav2VecEmbeds"]()
        wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        mathexpressionpysssss = NODE_CLASS_MAPPINGS["MathExpression|pysssss"]()

        for q in range(1):
            wanvideoextramodelselect_13 = wanvideoextramodelselect.getmodelpath(
                extra_model="Lynx/Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors",
                prev_model=get_value_at_index(wanvideoextramodelselect_69, 0),
            )

            wanvideomodelloader_12 = wanvideomodelloader.loadmodel(
                model="Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                base_precision="fp16_fast",
                quantization="disabled",
                load_device="offload_device",
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
            width=832, height=480, num_frames=get_value_at_index(mathexpressionpysssss_98, 0)
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
                num_frames=get_value_at_index(mathexpressionpysssss_98, 0),
                fps=24,
                audio_scale=1,
                audio_cfg_scale=2,
                multi_audio_type="para",
                wav2vec_model=get_value_at_index(downloadandloadwav2vecmodel_96, 0),
                audio_1=get_value_at_index(vhs_loadaudioupload_93, 0),
            )
            wanvideosampler_22 = wanvideosampler.process(
                steps=6,
                cfg=1,
                shift=8,
                seed=random.randint(1, 2**64),
                force_offload=True,
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
                enable_vae_tiling=False,
                tile_x=272,
                tile_y=272,
                tile_stride_x=144,
                tile_stride_y=128,
                normalization="default",
                vae=get_value_at_index(wanvideovaeloader_31, 0),
                samples=get_value_at_index(wanvideosampler_22, 0),
            )

            vhs_videocombine_77 = vhs_videocombine.combine_video(
                frame_rate=24,
                loop_count=0,
                filename_prefix=f"output_{output_suffix}",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(wanvideodecode_32, 0),
                audio=get_value_at_index(vhs_loadaudioupload_93, 0),
                unique_id=15650465383331186117,
            )

def handler(input):
    prompt = input["input"].get("prompt")
    image_url = input["input"].get("image_url")
    audio_url = input["input"].get("audio_url")
    
    # Create the inputs directory if it doesn't exist
    inputs_dir = "/root/comfy/comfyUI/inputs/"
    os.makedirs(inputs_dir, exist_ok=True)
    
    # Download image
    image_response = requests.get(image_url)
    image_response.raise_for_status()
    
    # Get file extension from URL or default to .jpg for image
    image_ext = os.path.splitext(image_url.split('?')[0])[1] or '.jpg'
    image_filename = f"{uuid.uuid4().hex}{image_ext}"
    image_path = os.path.join(inputs_dir, image_filename)
    
    with open(image_path, 'wb') as f:
        f.write(image_response.content)
    
    # Download audio
    audio_response = requests.get(audio_url)
    audio_response.raise_for_status()
    
    # Get file extension from URL or default to .wav for audio
    audio_ext = os.path.splitext(audio_url.split('?')[0])[1] or '.wav'
    audio_filename = f"{uuid.uuid4().hex}{audio_ext}"
    audio_path = os.path.join(inputs_dir, audio_filename)
    
    with open(audio_path, 'wb') as f:
        f.write(audio_response.content)
    
    random_suffix = uuid.uuid4().hex[:6]
    # Run the workflow with the downloaded files
    workflow(prompt, audio_path, image_path,random_suffix)
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
    else:
        output_path = None
    # Upload output video to S3 and get URL
    if output_path and os.path.exists(output_path):
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name='eu-north-1'  
        )
        
        bucket_name = 'fritz-comfyui'  
        s3_key = f"{output_filename}"
        
        try:
            # Upload file to S3
            s3_client.upload_file(output_path, bucket_name, s3_key)
            
            # Generate S3 URL
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
            
            output_s3_url = s3_url
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            output_s3_url = None
    else:
        output_s3_url = None
    return { 
        "message": "Success! Download the video from the provided URL.",
        "video_url": output_s3_url
    }
 
runpod.serverless.start({"handler": handler}) 
