# -*- coding: utf-8 -*-
import os
from modelscope import snapshot_download
d = snapshot_download(
    "modelscope/Llama-2-7b-chat-ms",
    local_dir=os.path.expanduser("~/rog_ft/llama2-7b-chat"),
    ignore_file_pattern=[r".*\.bin$", r".*\.pth$", r".*\.h5$", r".*\.msgpack$"],
)
print("DOWNLOADED to:", d)
import glob
for f in sorted(glob.glob(d + "/*")):
    print(" ", os.path.basename(f), round(os.path.getsize(f) / 1e9, 2), "GB")
