#!/usr/bin/env python3
"""
Script to trim the first 1 second from a video file.
Overwrites the output file if it already exists.
Safely handles the case where input and output are the same file.
"""

import subprocess
import sys
import os
import argparse
import tempfile
import shutil


def trim_video(input_path: str, output_path: str) -> None:
    """
    Trim the first 1 second from a video file.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to the output video file (will be overwritten if exists)
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        subprocess.CalledProcessError: If ffmpeg command fails
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if input and output are the same file
    same_file = os.path.abspath(input_path) == os.path.abspath(output_path)
    
    # If they're the same, use a temporary file
    if same_file:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        actual_output = temp_path
    else:
        actual_output = output_path
    
    try:
        print(f"Step 1: Trimming first 1 second from video...")
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-ss', '1',  # Start from 1 second (before -i for faster seeking)
            '-i', input_path,
            '-c:v', 'libx264',  # Re-encode video to fix keyframes
            '-preset', 'fast',  # Speed up encoding
            '-c:a', 'aac',  # Re-encode audio to AAC
            actual_output
        ], check=True)
        print(f"  ✓ Video trimmed: {actual_output}")
        
        # If input and output were the same, replace the original
        if same_file:
            print(f"Step 2: Replacing original file...")
            shutil.move(actual_output, output_path)
            print(f"  ✓ Original file updated")
        
        print(f"✓ Successfully trimmed first 1 second from video")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        
    except Exception as e:
        # Clean up temporary file if it exists
        if same_file and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e
 