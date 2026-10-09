"""Run a local Rerun server hosting the experiment datasets until Ctrl-C.

    python -m cfnbc.serve [--port 51234]

Datasets: every non-empty <rerun_root>/{demos,rollouts,training} dir is registered as cube_{name};
extra layers (qa, scene3d, counterfactual, rollout qa, ...) are registered per cfnbc.hub.LAYER_DIRS.
Tables are the Lance tables under <rerun_root>/tables/. Everything is re-registered from files on every start, so the
files under <rerun_root> are the source of truth. Leave it running, point the tools at it with
`export CFNBC_CATALOG_URL=rerun+http://127.0.0.1:51234`, and open the same URL in the viewer.
"""
from __future__ import annotations

import argparse
import time

import rerun as rr

from cfnbc import schema as S

DIRS = {S.DATASET_DEMOS: "demos", S.DATASET_ROLLOUTS: "rollouts", S.DATASET_TRAINING: "training",
        "real_demos": "real_demos"}


def dataset_dirs(root=None) -> dict[str, str]:
    root = root or S.rerun_root()
    return {name: str(root / sub) for name, sub in DIRS.items() if any((root / sub).glob("*.rrd"))}


def table_dirs(root=None) -> dict[str, str]:
    """Lance tables created by cfnbc.hub.ensure_table: <root>/tables/<name>/<unique id>/."""
    root = root or S.rerun_root()
    out = {}
    for d in sorted((root / "tables").glob("*/*")):
        if (d / "_versions").is_dir():
            out[d.parent.name] = str(d)
    return out


def start(port: int | None = None, root=None, layers: bool = True) -> rr.server.Server:
    """Start the server over <root>'s dataset dirs and register extra cube_demos layers (see cfnbc.hub.local_files)."""
    from cfnbc.hub import local_files
    try:  # thousands of layer files: lift the default 256 open-file limit (macOS) to the hard limit
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536, hard))
    except (ValueError, OSError):
        pass
    root = root or S.rerun_root()
    srv = rr.server.Server(port=port, datasets=dataset_dirs(root), tables=table_dirs(root))
    if layers:
        client = srv.client()
        for (dataset, layer), files in local_files(root).items():
            if layer == "base" or dataset not in client.dataset_names():
                continue
            uris = [f"file://{p.resolve()}" for p in files]
            client.get_dataset(dataset).register(uris, layer_name=layer).wait()
        register_blueprints(client)
    return srv


def register_blueprints(client, uri_for=lambda p: f"file://{p.resolve()}") -> None:
    """(Re)generate <rerun_root>/blueprints/<dataset>.rbl and set each as its dataset's default blueprint.

    The catalog references the registered file in place, so register an immutable, content-addressed copy: re-saving
    the working .rbl later must not change what is already registered.
    """
    import hashlib
    import shutil
    from cfnbc.blueprints import OUT, path_for, save_all
    save_all()
    for name in client.dataset_names():
        p = path_for(name)
        if not p.exists():
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        frozen = OUT / "registered" / f"{name}_{digest}.rbl"
        if not frozen.exists():
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, frozen)
        client.get_dataset(name).register_blueprint(uri_for(frozen), set_default=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=51234)
    args = parser.parse_args()
    srv = start(args.port)
    client = srv.client()
    print(f"Serving {client.dataset_names()} at {srv.url()}  (root {S.rerun_root()})")
    for ds in client.datasets():
        print(f"  {ds.name}: {len(ds.segment_ids())} segments")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
