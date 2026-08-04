import importlib
from pathlib import Path

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def scan_files_from_folder(dir_paths, exts, recursive=True):
    exts = [exts, ] if isinstance(exts, str) else exts
    dir_paths = [dir_paths, ] if isinstance(dir_paths, str) else dir_paths

    file_paths = []
    for current_dir in dir_paths:
        for current_ext in exts:
            if recursive:
                search_flag = f"**/*.{current_ext}"
            else:
                search_flag = f"*.{current_ext}"
            file_paths.extend(sorted([str(x) for x in Path(current_dir).glob(search_flag)]))

    return file_paths
