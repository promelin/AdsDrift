"""AdsDrift web dashboard — demonstrates the project's working components.

This is a lightweight Flask app that wraps the existing AdsDrift codebase.
It does NOT modify any project functionality; it only imports and exercises
the parts that work without external data (adsorbate library, R0 random
structure sampling, generator architecture inspection).
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# AdsDrift package setup — the repo root must be importable as "AdsDrift".
# In Docker the repo is mounted at /workspace/AdsDrift with PYTHONPATH=/workspace.
# Locally the directory may have a different name, so we create a symlink.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE))

_adsdrift_link = WORKSPACE / "AdsDrift"
if REPO_ROOT.name != "AdsDrift" and not _adsdrift_link.exists():
    try:
        _adsdrift_link.symlink_to(REPO_ROOT)
    except OSError:
        pass

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ---------------------------------------------------------------------------
# Demo surfaces for the R0 sampling demo
# ---------------------------------------------------------------------------
DEMO_SURFACES = {
    "Cu(111)": {"element": "Cu", "crystal": "fcc", "a": 3.6, "miller": [1, 1, 1]},
    "Pt(111)": {"element": "Pt", "crystal": "fcc", "a": 3.92, "miller": [1, 1, 1]},
    "Ni(111)": {"element": "Ni", "crystal": "fcc", "a": 3.52, "miller": [1, 1, 1]},
    "Cu(100)": {"element": "Cu", "crystal": "fcc", "a": 3.6, "miller": [1, 0, 0]},
}


def _load_adsorbate_entries():
    """Return a list of adsorbate library entries for display."""
    from AdsDrift.generate.adsorbate_library import read_library, DEFAULT_LIBRARY

    lib = read_library(DEFAULT_LIBRARY)
    entries = []
    for key, entry in lib["adsorbates"].items():
        entries.append(
            {
                "id": key,
                "name": entry["name"],
                "formula": entry["formula"],
                "atom_count": len(entry["atomic_numbers"]),
                "binding_indices": entry.get("binding_indices", []),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Model cache (instantiation is expensive — load once per config)
# ---------------------------------------------------------------------------
_model_cache: dict = {"config_name": None, "model": None, "config": None}


def _get_model(config_name: str):
    import yaml

    from AdsDrift.model.generator import build_generator, EquiformerV3GeneratorConfig

    if _model_cache["config_name"] != config_name:
        config_path = REPO_ROOT / "config" / config_name
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        gen_config = full_config["generator"]
        model = build_generator(gen_config)
        EquiformerV3GeneratorConfig.from_dict(gen_config)  # validate
        _model_cache.update(
            {"config_name": config_name, "model": model, "config": gen_config}
        )
    return _model_cache["model"], _model_cache["config"]


def _list_config_files():
    config_dir = REPO_ROOT / "config"
    return sorted(p.name for p in config_dir.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/adsorbates")
def adsorbates():
    entries = _load_adsorbate_entries()
    return render_template("adsorbates.html", adsorbates=entries, count=len(entries))


@app.route("/sampling")
def sampling():
    entries = _load_adsorbate_entries()
    return render_template(
        "sampling.html", surfaces=DEMO_SURFACES, adsorbates=entries
    )


@app.route("/api/sample", methods=["POST"])
def api_sample():
    data = request.get_json(silent=True) or {}
    surface_name = data.get("surface", "Cu(111)")
    adsorbate_id = str(data.get("adsorbate", "5"))
    num_samples = int(data.get("num_samples", 100))
    seed = int(data.get("seed", 42))

    surface_config = DEMO_SURFACES.get(surface_name, DEMO_SURFACES["Cu(111)"])

    try:
        from ase.build import bulk
        from ase.io import write

        from AdsDrift.generate.structure_input import combine_reference, prepare_slab
        from AdsDrift.generate.adsorbate_library import get_adsorbate
        from AdsDrift.model.initialize.generate_random_inputs import (
            coverage_report,
            generate_batch,
            write_coverage_plot,
        )

        crystal = bulk(
            surface_config["element"],
            surface_config["crystal"],
            a=surface_config["a"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bulk_path = Path(tmpdir) / "bulk.xyz"
            write(str(bulk_path), crystal)

            slab, slab_meta = prepare_slab(
                bulk_path,
                input_kind="bulk",
                miller=surface_config["miller"],
                layers=4,
                repeat=(2, 2),
                vacuum=15.0,
                fixed_layers=2,
            )

            molecule, ads_meta = get_adsorbate(adsorbate_id)
            reference = combine_reference(slab, molecule, gap_max=3.0)

            batch, report = generate_batch(
                reference,
                count=num_samples,
                seed=seed,
                system_id=f"{surface_name}_{adsorbate_id}",
            )

            coverage = coverage_report(
                batch, grid_size=10, dedup_tolerance=0.1
            )

            plot_path = Path(tmpdir) / "coverage.png"
            write_coverage_plot(batch, coverage, plot_path)

            with open(plot_path, "rb") as f:
                plot_b64 = base64.b64encode(f.read()).decode("ascii")

        return jsonify(
            {
                "plot": f"data:image/png;base64,{plot_b64}",
                "surface": surface_name,
                "adsorbate": ads_meta["name"],
                "adsorbate_formula": ads_meta["formula"],
                "slab_atoms": slab_meta["slab_atom_count"],
                "adsorbate_atoms": len(molecule),
                "total_atoms": len(reference),
                "num_samples": report["count"],
                "unique_positions": coverage[
                    "periodic_unique_lateral_positions"
                ],
                "grid_coverage": coverage[
                    "grid_coverage_fraction_after_dedup"
                ],
                "occupied_cells": coverage[
                    "occupied_grid_cells_after_dedup"
                ],
                "total_cells": coverage["total_grid_cells"],
                "min_distance": coverage[
                    "minimum_pairwise_lateral_distance_angstrom"
                ],
                "rejected_contact": report["rejected_contact"],
                "rejected_vacuum": report["rejected_vacuum"],
                "rejected_lateral": report["rejected_lateral_duplicate"],
                "proposals": report["proposal_count"],
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/model")
def model_info():
    config_name = request.args.get("config", "train_single_0_1190_0.yaml")
    config_files = _list_config_files()

    try:
        model, gen_config = _get_model(config_name)
        param_count = model.parameter_count
        error = None
    except Exception as exc:
        param_count = None
        gen_config = {}
        error = str(exc)

    return render_template(
        "model_info.html",
        config_name=config_name,
        config_files=config_files,
        gen_config=gen_config,
        param_count=param_count,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
