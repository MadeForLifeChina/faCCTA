output_root = '/data/xxx/ProductOutput/faCCTA'


data_path = {
    "image": r"/data/fast_storage/xxx/RawData/CoronaryArtery/NIFTI//Image",
    "gt_artery": r"/data/fast_storage/xxx/RawData/CoronaryArtery/NIFTI/GT_AortaLumen",
    "gt_heart": r"/data/fast_storage/xxx/RawData/CoronaryArtery/NIFTI/GT_Heart",
    "gt_plaque": r"/data/fast_storage/xxx/RawData/CoronaryArtery/NIFTI/GT_Plaque",
}

gpu_ids = {
    "Task_10001_Heart": [0],
    "Task_10002_Coarse": [0],
    "Task_10003_Fine": [0],
}

path_size = {
    "Task_10001_Heart": (160, 160, 160),
    "Task_10002_Coarse": (160, 160, 160),
    "Task_10003_Fine": (160, 160, 160),
}

spacing = {
    "Task_10001_Heart": None,
    "Task_10002_Coarse": (0.8, 0.8, 0.8),
    "Task_10003_Fine": (0.4, 0.4, 0.4),
}

extend = {
    "Task_10003_Fine": (10., 10., 10., 10., 10., 10.),
}


