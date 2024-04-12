from huggingface_hub import hf_hub_download

# The dataset consists of 35 zip files split into 5 groups (7 zip files per group)
for i in range(1):
    for j in range(7):
        hf_hub_download(repo_id="FoivosPar/Arc2Face", filename=f"{i}/{i}_{j}.zip", 
                        local_dir="/data/shaohua/Arc2Face_data", repo_type="dataset",
                        local_dir_use_symlinks=False)
