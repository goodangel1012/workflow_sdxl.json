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
    
    silence_path = f"{output_path}_silence.wav"
    input_converted = f"{output_path}_input_converted.wav"
    concat_list_path = f"{output_path}_concat.txt"
    
    try:
        # Step 1: Create 2 second silence as WAV
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
        
        # Step 2: Convert input audio to same format as silence
        print("Step 2: Converting input audio to matching format...")
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', input_path,
            '-q:a', '9',
            '-acodec', 'libmp3lame',
            input_converted
        ], check=True)
        print(f"  ✓ Input converted: {input_converted}")
        
        # Step 3: Concatenate silence + converted audio
        print("Step 3: Concatenating silence with audio...")
        with open(concat_list_path, 'w') as f:
            f.write(f"file '{silence_path}'\n")
            f.write(f"file '{input_converted}'\n")
        
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
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
    input_path = "input.wav"
    output_path = "input.wav"
    add_silence_to_audio(input_path, output_path)


if __name__ == '__main__':
    main()
