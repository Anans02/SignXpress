#!/usr/bin/env python3
"""
Setup Script for SignXpress
Installs required dependencies
"""

import subprocess
import sys
import os

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def main():
    print("=== SignXpress Setup ===")
    
    # Create virtual environment
    print("Creating virtual environment...")
    if not run_command("python -m venv signxpress_env"):
        print("Failed to create virtual environment")
        return
    
    # Install requirements
    print("Installing dependencies...")
    
    if sys.platform == "win32":
        pip_path = "signxpress_env\\Scripts\\pip"
        activate_cmd = "signxpress_env\\Scripts\\activate"
    else:
        pip_path = "signxpress_env/bin/pip"
        activate_cmd = "source signxpress_env/bin/activate"
    
    requirements = [
        "tensorflow",
        "opencv-python", 
        "mediapipe",
        "numpy",
        "pandas"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        run_command(f"{pip_path} install {package}")
    
    print("\n=== SETUP COMPLETE ===")
    print("Next steps:")
    print(f"1. Activate environment: {activate_cmd}")
    print("2. Run your application!")

if __name__ == "__main__":
    main()