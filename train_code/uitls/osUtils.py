import os
import shutil


def count_subdirectories(path):
    if not os.path.isdir(path):
        raise ValueError("The provided path is not a directory.")

    subdirectories = [
        entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))
    ]
    return len(subdirectories)


def copy_tree(fromDir, TargetDir):
    if len(os.listdir(TargetDir)) > 0:
        return
    for item in os.listdir(fromDir):
        src_item = os.path.join(fromDir, item)
        dst_item = os.path.join(TargetDir, item)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, symlinks=False, ignore=None)
        else:
            shutil.copy2(src_item, dst_item)  # 使用 copy2 以保留文件元数据
