import numpy as np
from scipy.io import loadmat

mat_path = "/mnt/dataset2/Datasets/EAV/EAV/subject1/EEG/subject1_eeg.mat"

# squeeze_me=True 让多余维度更好读；struct_as_record=False 更容易访问struct
data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

# 过滤掉Matlab的元信息变量
keys = [k for k in data.keys() if not k.startswith("__")]
print("Variables:", keys)

def pretty(x, max_items=10, indent=0):
    pad = " " * indent
    if isinstance(x, np.ndarray):
        print(f"{pad}ndarray shape={x.shape}, dtype={x.dtype}")
        # 打印少量内容预览
        flat = x.ravel()
        preview = flat[:max_items]
        print(f"{pad}preview: {preview}{' ...' if flat.size > max_items else ''}")
    elif hasattr(x, "_fieldnames"):  # MATLAB struct（scipy会变成 mat_struct）
        print(f"{pad}struct fields={x._fieldnames}")
        for f in x._fieldnames:
            print(f"{pad}{f} =")
            pretty(getattr(x, f), max_items=max_items, indent=indent+2)
    else:
        print(f"{pad}{type(x)}: {x}")

for k in keys:
    print("\n==", k, "==")
    pretty(data[k])