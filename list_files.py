from huggingface_hub import list_repo_files
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
files = list_repo_files(repo_id="deepfakesMSU/NTIRE-RobustAIGenDetection-val", repo_type="dataset")
print(files)
