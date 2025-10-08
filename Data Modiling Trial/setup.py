import os
import subprocess
import sys
from pathlib import Path


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Enter running: {command}")
        print(e)
        sys.exit(1)
        
def main():
    print("Setting up project")

    print("Creating virtual environment...")
    run_command("pythin -m venv venv")

    print("Activating virtual environment")
    run_command("venv\\Scripts\\activate")
    
    print("Installing libraries")
    run_command("pip install -r requriements.txt")

    print("Creating folders...")
    folders = ["src", "src/models", "tests", "data"]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        
    print("Setup don! Run 'venv\\Scripts\\activate' then 'pyhton main.py' to start.")
    
if __name__ as "__main__":
    main()