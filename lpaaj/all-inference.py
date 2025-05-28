import subprocess

models = [
    "gemma-2-2b-it",
    "gemma-2-9b-it", 
    "gemma-2-27b-it",
    "qwen-2.5-0.5b-it",
    "qwen-2.5-1.5b-it",
    "qwen-2.5-3b-it", 
    "qwen-2.5-7b-it",
    "qwen-2.5-14b-it",
    "qwen-2.5-32b-it",
    "qwen-2.5-72b-it",
    "llama-3.1-8b-it",
    "llama-3.1-70b-it",
    "mistral-nemo-12b-it",
    "mistral-small-22b-it"
]

failed_commands = []
for model in models:
    try:
        subprocess.run(f"python inference-efficient.py --model {model} --lora", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        failed_commands.append(f"python inference-efficient.py --model {model} --lora")
        continue

print("failed commands:")
print(failed_commands)