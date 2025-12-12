import os
import cartesia
from pathlib import Path
from typing import Optional
import subprocess
import tempfile

API_KEY="sk_car_VFRsm464A7ZBxWTTX5cRbF"
def generate_audio_from_transcript(
    transcript: str,
    output_filename: str,
    api_key: Optional[str] = API_KEY,
    voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",  # Default Cartesia voice
    model_id: str = "sonic-english",
    output_format: str = "mp3",
    language: str = "en"
) -> str:
    """
    Generate audio from a transcript using Cartesia TTS and save it to the audios/ directory.
    
    Args:
        transcript (str): The text to convert to speech
        output_filename (str): The filename for the output audio file (without extension)
        api_key (str, optional): Cartesia API key. If None, reads from CARTESIA_API_KEY env var
        voice_id (str): Cartesia voice ID to use for generation
        model_id (str): Cartesia model ID to use
        output_format (str): Output audio format (mp3, wav, raw)
        language (str): Language code for the transcript
    
    Returns:
        str: Path to the generated audio file
    
    Raises:
        ValueError: If API key is not provided or found in environment
        Exception: If audio generation fails
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("CARTESIA_API_KEY")
    
    if not api_key:
        raise ValueError("API key must be provided either as parameter or CARTESIA_API_KEY environment variable")
    
    # Create audios directory if it doesn't exist
    audios_dir = Path("audios")
    audios_dir.mkdir(exist_ok=True)
    
    # Prepare output path
    output_path = audios_dir / f"{output_filename}.{output_format}"
    
    try:
        # Initialize Cartesia client
        client = cartesia.Cartesia(api_key=api_key)
        
        # Prepare voice parameter
        voice = {"mode": "id", "id": voice_id}
        
        # Prepare output format parameter based on format type
        if output_format.lower() == "mp3":
            output_format_param = {
                "container": "mp3",
                "encoding": "mp3",
                "sample_rate": 44100,
            }
        elif output_format.lower() == "wav":
            output_format_param = {
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            }
        else:  # raw
            output_format_param = {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            }
        
        # Generate audio using the bytes method
        response = client.tts.bytes(
            model_id=model_id,
            transcript=transcript,
            voice=voice,
            output_format=output_format_param,
            language=language,
            
        )
        
        # Save audio to temporary file first
        temp_path = audios_dir / f"{output_filename}_temp.{output_format}"
        with open(temp_path, "wb") as audio_file:
            for chunk in response:
                audio_file.write(chunk)
        
        # Use FFmpeg to speed up the audio by 1.5x
        try:
            subprocess.run([
                'ffmpeg', '-i', str(temp_path), 
                '-filter:a', 'atempo=1.3', 
                '-y', str(output_path)
            ], check=True, capture_output=True)
            
            # Remove temporary file
            temp_path.unlink()
            
        except subprocess.CalledProcessError as e:
            # If FFmpeg fails, fallback to original file
            print(f"Warning: FFmpeg processing failed ({e}), using original audio")
            temp_path.rename(output_path)
        except FileNotFoundError:
            # If FFmpeg is not installed, fallback to original file
            print("Warning: FFmpeg not found, using original audio without speed adjustment")
            temp_path.rename(output_path)
        
        print(f"Audio successfully generated and saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        raise Exception(f"Failed to generate audio: {str(e)}")


def list_available_voices(api_key: Optional[str] = API_KEY) -> list:
    """
    List available voices from Cartesia.
    
    Args:
        api_key (str, optional): Cartesia API key
    
    Returns:
        list: List of available voices
    """
    if api_key is None:
        api_key = os.getenv("CARTESIA_API_KEY")
    
    if not api_key:
        raise ValueError("API key must be provided either as parameter or CARTESIA_API_KEY environment variable")
    
    try:
        client = cartesia.Cartesia(api_key=api_key)
        voices = client.voices.list()
        return voices
    except Exception as e:
        raise Exception(f"Failed to fetch voices: {str(e)}")
 