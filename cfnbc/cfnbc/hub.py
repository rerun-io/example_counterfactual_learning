"""Catalog access that works the same against the open-source Rerun server and the Rerun Hub.

    CFNBC_CATALOG_URL      catalog to use: the server started by `python -m cfnbc.serve` (rerun+http://127.0.0.1:51234)
                           or a Rerun Hub. Unset -> start a throwaway in-process server over <rerun_root>.
    CFNBC_HUB              set to 1 when CFNBC_CATALOG_URL is a Rerun Hub (the Hub picks table locations itself).
    CFNBC_CATALOG_TOKEN    optional auth token for CFNBC_CATALOG_URL.
    CFNBC_STORAGE_PREFIX   where the catalog reads .rrd files from (file:///... or s3://...). Unset -> the files'
                           own file:// paths under <rerun_root>.

    with hub.open_catalog() as client:
        ds = client.get_dataset("cube_demos")

    python -m cfnbc.hub sync [--datasets cube_demos ...] [--replace]   # register rerun/ files with the catalog
    python -m cfnbc.hub info
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
from pathlib import Path
from typing import Iterator

import rerun as rr
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer

from cfnbc import schema as S
from cfnbc.serve import DIRS

# extra layers: <rerun_root>/<subdir>/*.rrd -> (dataset, fixed layer name), or
#               <rerun_root>/<subdir>/<layer>/*.rrd -> (dataset, None: layer = directory name)
LAYER_DIRS = {
    "counterfactual": (S.DATASET_DEMOS, None),
    "qa": (S.DATASET_DEMOS, "qa"),
    "scene3d": (S.DATASET_DEMOS, "scene3d"),
    "scene3d_nuisance": (S.DATASET_DEMOS, None),
    "ghost": (S.DATASET_DEMOS, None),
    "rollout_qa": (S.DATASET_ROLLOUTS, "qa"),
    "rollout_scene3d": (S.DATASET_ROLLOUTS, "scene3d"),
    "train_pad": (S.DATASET_DEMOS, "train_pad"),
    "real_provenance": ("real_demos", "provenance"),
    "real_qa": ("real_demos", "qa"),
    "real_masks": ("real_demos", "masks"),
    "real_counterfactual": ("real_demos", None),
    "real_train_pad": ("real_demos", "train_pad"),
}



def catalog_url() -> str | None:
    return os.environ.get("CFNBC_CATALOG_URL") or None


@contextlib.contextmanager
def open_catalog(root: Path | None = None) -> Iterator[CatalogClient]:
    """Yield a CatalogClient. Keep using it inside the block (lazy DataFrames need the client alive)."""
    url = catalog_url()
    if url:
        yield CatalogClient(url, token=os.environ.get("CFNBC_CATALOG_TOKEN"))
        return
    from cfnbc.serve import start
    with start(port=0, root=root) as srv:
        yield srv.client()


def storage_uri(path: Path, root: Path | None = None) -> str:
    """URI under which the catalog can read `path` (a file under <rerun_root>)."""
    root = (root or S.rerun_root()).resolve()
    prefix = os.environ.get("CFNBC_STORAGE_PREFIX")
    path = Path(path).resolve()
    if not prefix:
        return f"file://{path}"
    return prefix.rstrip("/") + "/" + path.relative_to(root).as_posix()


def stage_to_storage(paths: list[Path], root: Path | None = None) -> set[Path]:
    """Copy files to a file:// storage prefix (no-op without a prefix); s3:// prefixes must be synced externally.

    Returns the subset of `paths` whose staged copy already existed and was overwritten: the catalog's registration of
    those is stale ("the source object changed; re-register it") and must be re-registered with REPLACE.
    """
    prefix = os.environ.get("CFNBC_STORAGE_PREFIX")
    if not prefix:
        return set()
    if not prefix.startswith("file://"):
        raise NotImplementedError(f"upload to {prefix} not implemented; sync <rerun_root> there (e.g. aws s3 sync)")
    root = (root or S.rerun_root()).resolve()
    dest_root = Path(prefix.removeprefix("file://"))
    changed = set()
    for p in paths:
        dest = dest_root / Path(p).resolve().relative_to(root)
        if not dest.exists() or dest.stat().st_mtime < Path(p).stat().st_mtime:
            if dest.exists():
                changed.add(Path(p))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    return changed


def tables_root(root: Path | None = None) -> Path:
    return (root or S.rerun_root()) / "tables"


def ensure_table(client: CatalogClient, name: str, schema, root: Path | None = None):
    """Get or create a Lance-backed table entry.

    On the open-source server the table is stored under <rerun_root>/tables/<name>/, where `cfnbc.serve` finds it again
    after a restart (without a url the server would put it in a temporary directory). A Hub manages table storage
    itself and rejects a caller-supplied url.
    """
    if name in client.table_names():
        return client.get_table(name)
    url = None
    if not on_hub():
        d = tables_root(root) / name
        d.mkdir(parents=True, exist_ok=True)
        url = f"file://{d.resolve()}"
    return client.create_table(name, schema, url=url)


def on_hub() -> bool:
    return bool(catalog_url()) and os.environ.get("CFNBC_HUB", "") not in ("", "0")


def ensure_dataset(client: CatalogClient, name: str):
    return client.create_dataset(name, exist_ok=True)


def register(client: CatalogClient, dataset: str, paths: list[Path], layer: str = "base", replace: bool = False,
             root: Path | None = None) -> list[str]:
    """Batch-register .rrd files (one call) as `layer`; returns registered segment ids."""
    if not paths:
        return []
    changed = stage_to_storage(paths, root)
    ds = ensure_dataset(client, dataset)
    mode = OnDuplicateSegmentLayer.REPLACE if replace else OnDuplicateSegmentLayer.SKIP
    res = ds.register([storage_uri(p, root) for p in paths], layer_name=layer, on_duplicate=mode).wait()
    ids = list(res.segment_ids)
    if changed and not replace:  # files rewritten since they were registered: refresh those registrations
        res = ds.register([storage_uri(p, root) for p in sorted(changed)], layer_name=layer,
                          on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        ids += list(res.segment_ids)
        print(f"  re-registered {len(changed)} rewritten file(s) in {dataset} [{layer}]")
    return ids


def local_files(root: Path | None = None) -> dict[tuple[str, str], list[Path]]:
    """{(dataset, layer): [rrd paths]} for everything under <rerun_root>."""
    root = root or S.rerun_root()
    out: dict[tuple[str, str], list[Path]] = {}
    for dataset, sub in DIRS.items():
        files = sorted((root / sub).glob("*.rrd"))
        if files:
            out[(dataset, "base")] = files
    for sub, (dataset, fixed_layer) in LAYER_DIRS.items():
        base = root / sub
        if not base.is_dir():
            continue
        if fixed_layer:
            files = sorted(base.glob("*.rrd"))
            if files:
                out[(dataset, fixed_layer)] = files
        else:
            for d in sorted(p for p in base.iterdir() if p.is_dir()):
                files = sorted(d.glob("*.rrd"))
                if files:
                    out[(dataset, d.name)] = files
    return out


def sync(datasets: list[str] | None = None, replace: bool = False, root: Path | None = None) -> None:
    if not catalog_url():
        raise SystemExit("CFNBC_CATALOG_URL not set; the in-process OSS server registers rerun/ automatically")
    with open_catalog(root) as client:
        for (dataset, layer), files in local_files(root).items():
            if datasets and dataset not in datasets:
                continue
            ids = register(client, dataset, files, layer=layer, replace=replace, root=root)
            print(f"{dataset} [{layer}]: {len(files)} files -> {len(ids)} segments")
        from cfnbc.serve import register_blueprints
        register_blueprints(client)  # blueprint files must be readable by the catalog server


def info() -> None:
    with open_catalog() as client:
        print(f"catalog: {catalog_url() or 'in-process OSS server'}  version: {client.version_info()}")
        for ds in client.datasets():
            st = ds.segment_table().to_pandas()
            layers = sorted({l for ls in st["rerun_layer_names"] for l in ls})
            print(f"  dataset {ds.name}: {len(st)} segments, layers {layers}")
        for t in client.tables():
            print(f"  table {t.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--datasets", nargs="*")
    p_sync.add_argument("--replace", action="store_true")
    sub.add_parser("info")
    args = parser.parse_args()
    if args.cmd == "sync":
        sync(args.datasets, args.replace)
    else:
        info()


if __name__ == "__main__":
    main()
