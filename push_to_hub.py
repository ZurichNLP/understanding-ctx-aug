from huggingface_hub import HfApi
api = HfApi()

repo_id = "tannonk/understanding-ctx-aug"
# api.create_repo(repo_id)

print(f"Created repo {repo_id}")

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
for seed in [23, 42, 1984]:
    model_dir = f"resources/models/seed_{seed}/ft/"
    repo_path = f"models/ft/topical-chat-s{seed}"
    print(f"Uploading {model_dir} to Hugging Face Hub")
    api.upload_folder(
        folder_path=model_dir,
        path_in_repo=repo_path,
        repo_id=repo_id,
    )
