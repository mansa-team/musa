from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="unsloth/Qwen3.5-4B-GGUF",
    filename="Qwen3.5-4B-Q4_K_M.gguf",
    local_dir="D:/Repositories/research/logun/models",
)
print("OK " + path)
