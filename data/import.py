from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="atomscott/soccertrack-v2",
    repo_type="dataset",
    filename="bas/118576/118576_12_class_events.json"
)
print("Downloaded to:", file_path)