import pickle
import subprocess
import json


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def run_powershell_script(script_path):
    # Executes the PowerShell script
    try:
        result = subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path], check=True, text=True, capture_output=True)
        print("Script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")