python -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='l3cube-pune/hindi-bert-v2',
    local_dir='models/hindi-bert-v2',
    local_dir_use_symlinks=False
)
"
