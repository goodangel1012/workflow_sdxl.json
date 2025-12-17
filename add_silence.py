#!/usr/bin/env python3
"""
Script to add 2 seconds of silence at the start of an audio file.
Overwrites the output file if it already exists.
"""

import subprocess
import sys
import os
import argparse


def add_silence_to_audio(input_path: str, output_path: str) -> None:
    """
    Add 2 seconds of silence at the start of an audio file.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to the output audio file (will be overwritten if exists)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    silence_path = f"{output_path}_silence.mp3"
    input_converted = f"{output_path}_input_converted.mp3"
    concat_list_path = f"{output_path}_concat.txt"
    
    try:
        # Step 1: Create 2 second silence as MP3
        print("Step 1: Creating 2 seconds of silence...")
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'lavfi',
            '-i', 'anullsrc=r=44100:cl=mono',
            '-t', '2',
            '-q:a', '9',
            '-acodec', 'libmp3lame',
            silence_path
        ], check=True)
        print(f"  ✓ Silence created: {silence_path}")
        
        # Step 2: Convert input audio to MP3
        print("Step 2: Converting input audio to MP3...")
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', input_path,
            '-q:a', '9',
            '-acodec', 'libmp3lame',
            input_converted
        ], check=True)
        print(f"  ✓ Input converted: {input_converted}")
        
        # Step 3: Concatenate using filter_complex (more reliable than concat demuxer)
        print("Step 3: Concatenating silence with audio...")
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', silence_path,
            '-i', input_converted,
            '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
            '-map', '[out]',
            '-q:a', '9',
            '-acodec', 'libmp3lame',
            output_path
        ], check=True)
        
        print(f"✓ Successfully added 2 seconds of silence to the start of audio")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        
    finally:
        # Clean up temporary files
        for temp_file in [silence_path, input_converted, concat_list_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {e}")


def main():
    input_path = "input.mp3"
    output_path = "output.mp3"
    add_silence_to_audio(input_path, output_path)


if __name__ == '__main__':
    main()
