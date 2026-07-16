from huggingface_hub import snapshot_download

data = snapshot_download( #Loading the data for training
                "HuggingFaceFW/fineweb",
                repo_type="dataset",
                local_dir="./fineweb/",
                allow_patterns="sample/100BT/*")



