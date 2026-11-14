"""Workspace menu bridge; image workspaces retain their own code and jobs."""
from pathlib import Path


def change_workspace(root, plain=False):
    from image.entry import SOURCE, choose_workspace, launch, matches_source
    from image.workspace import MANIFEST, PREFERENCE
    root = Path(root).resolve()
    if not (SOURCE / MANIFEST).is_file() or not matches_source(SOURCE, root):
        raise ValueError('当前是源码开发目录或其他镜像版本。源码 ./quickstart 的存储位置就是该源码目录；'
                         '请使用独立源码副本并在其中运行 ./quickstart。'
                         '镜像用户请运行全局命令 quickstart workspace 选择新目录；不会移动已有结果。')
    destination = choose_workspace(SOURCE, root, Path.home() / PREFERENCE, current=root, plain=plain)
    if destination is not None:
        launch(destination, ['--plain'] if plain else [])
