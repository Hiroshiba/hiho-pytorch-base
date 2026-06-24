"""pathlistファイル処理ユーティリティ"""

from upath import UPath

from .file_cache import RemoteFileCache

PathMap = dict[str, UPath]
"""パスマップ。stemをキー、パスを値とする辞書型"""


def _load_pathlist(
    pathlist_path: UPath, root_dir: UPath, file_cache: RemoteFileCache
) -> PathMap:
    """pathlistファイルを読み込んでパスマップを返す。"""
    path_list = [
        root_dir / p
        for p in file_cache.download(pathlist_path).read_text().splitlines()
    ]
    return {p.stem: p for p in path_list}


def get_data_paths(
    root_dir: UPath | None,
    pathlist_paths: list[UPath],
    file_cache: RemoteFileCache,
) -> tuple[list[str], list[PathMap]]:
    """複数のpathlistファイルからstemリストとパスマップを返す。整合性も確認する。"""
    if len(pathlist_paths) == 0:
        raise ValueError("少なくとも1つのpathlist設定が必要です")

    if root_dir is None:
        root_dir = UPath(".")

    path_mappings: list[PathMap] = []

    first_pathlist_path = pathlist_paths[0]
    first_paths = _load_pathlist(first_pathlist_path, root_dir, file_cache)
    fn_list = list(first_paths.keys())
    assert len(fn_list) > 0, f"ファイルが存在しません: {first_pathlist_path}"

    path_mappings.append(first_paths)

    for pathlist_path in pathlist_paths[1:]:
        paths = _load_pathlist(pathlist_path, root_dir, file_cache)
        assert fn_list == list(paths.keys()), (
            f"ファイルが一致しません: {pathlist_path} (expected: {len(fn_list)}, got: {len(paths)})"
        )
        path_mappings.append(paths)

    return fn_list, path_mappings
