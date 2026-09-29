"""Read recovered results without inventing per-problem data or standard errors."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid constant {x}")))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(root: Path, name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Source path must stay inside the reference directory")
    if any((root / q).is_symlink() for q in (p, *p.parents)):
        raise ValueError("Reference sources must not be symlinks")
    return root / p


def load_sources(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "historical_reference_results":
        raise ValueError("Not a historical reference manifest")
    expected = {"math_sft_passn", "math_sft_majority", "math_rl_passn", "math_rl_majority",
                "protein_unconditional", "protein_conditional", "majority_diagonal", "bon_diagonal"}
    if set(manifest["sources"]) != expected:
        raise ValueError("Missing or unknown reference source")
    sources = {}
    for key, item in manifest["sources"].items():
        path = _relative(root, item["file"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {key}")
        if path.suffix == ".json":
            sources[key] = read_json(path)
        elif path.suffix == ".csv":
            sources[key] = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        else:
            raise ValueError("Unsupported reference format")
    if manifest["sources"]["majority_diagonal"].get("statistic") != "a/(a+C)":
        raise ValueError("Figure 9 conversion requires the original plus-denominator statistic")
    if manifest["sources"]["bon_diagonal"].get("statistic") != "a/(a-C)":
        raise ValueError("Figure 10 is already a minus-denominator statistic")
    return manifest, sources


def _curve(values, probability=True):
    curve = {int(k): float(v) for k, v in values.items()} if isinstance(values, dict) else {
        i + 1: float(v) for i, v in enumerate(values)}
    if not curve or any(k < 1 or not math.isfinite(v) or
                        (probability and not 0 <= v <= 1) for k, v in curve.items()):
        raise ValueError("Invalid saved curve")
    return dict(sorted(curve.items()))


def curves(sources):
    result = {}
    for family, id_key, value_key in [("math_sft_passn", "model", "pass_at_k"),
                                       ("math_sft_majority", "strategy", "maj_at_k")]:
        entries = sources[family]
        if len({entry[id_key] for entry in entries}) != len(entries):
            raise ValueError("Duplicate saved model")
        result[family] = {entry[id_key]: _curve(entry[value_key]) for entry in entries}
    result["math_rl_passn"] = {key: _curve(value) for key, value in sources["math_rl_passn"].items()}
    rows = sources["math_rl_majority"]
    if len({r["Model"] for r in rows}) != len(rows):
        raise ValueError("Duplicate Majority Vote RL model")
    result["math_rl_majority"] = {row["Model"]: _curve({k: row[f"Maj@{k}" if k != 1 else "Pass@1"]
                                                     for k in (1, 4, 8, 16)}) for row in rows}
    for family, field in [("protein_unconditional", "curve"), ("protein_conditional", "scaling")]:
        result[family] = {name: _curve(value[field], probability=False)
                          for name, value in sources[family].items()}
    return result


def corrected_plus_ratio(value):
    """Convert the *original* a/(a+C) statistic once; never clip or drop a row."""
    value = float(value)
    if not math.isfinite(value) or not .5 < value <= 1:
        raise ValueError("Original plus-denominator ratio must lie in (0.5, 1]")
    return value / (2 * value - 1)


def diagnostic_rows(sources, kind):
    if kind not in {"majority_diagonal", "bon_diagonal"}:
        raise ValueError("Unknown diagnostic")
    rows = []
    for row in sources[kind]:
        p, original = float(row["p_val"]), float(row["rho"])
        if not math.isfinite(p) or not 0 < p < 1 or not math.isfinite(original):
            raise ValueError("Invalid diagnostic input")
        value = corrected_plus_ratio(original) if kind == "majority_diagonal" else original
        if value < 1:
            raise ValueError("Saved corrected ratio falls below one; do not silently clip")
        rows.append({"p_val": p, "rho_original": original, "rho": value})
    if not rows:
        raise ValueError("Empty diagnostic")
    return rows


def reference_audit(root):
    root = Path(root)
    manifest, source = load_sources(root)
    values = curves(source)
    checks = read_json(root / "paper_mean_checks.json")["checks"]
    comparison = []
    for item in checks:
        family = item["group"]
        raw = values[family][item["model"]][item["budget"]]
        scale = 100 if family.startswith("math") else 1
        places = 1 if family.startswith("math") else 2
        delta = raw * scale - item["reported_mean"]
        comparison.append({**item, "saved_mean": raw, "saved_in_paper_units": raw * scale,
                           "difference": delta, "matches_display": abs(delta) <= .5 * 10**(-places) + 1e-10})
    diagnostic = {}
    for key in ("majority_diagonal", "bon_diagonal"):
        rows = diagnostic_rows(source, key)
        rho = [r["rho"] for r in rows]
        diagnostic[key] = {"count": len(rows), "mean": mean(rho), "median": median(rho),
                           "min": min(rho), "max": max(rho)}
    return {"kind": "saved_result_audit_not_training_reproduction", "sources_verified": len(source),
            "checks": comparison, "display_matches": sum(x["matches_display"] for x in comparison),
            "display_comparisons": len(comparison), "diagnostics": diagnostic,
            "standard_errors_recomputed": False}


TABLES = (
    ("table2_passn_sft", "math_sft_passn", "tab:pass_results", (1, 4, 8, 16, 32, 64),
     (("SFT_Baseline", "SFT baseline"), ("PassAtN_4", "Pass@4"), ("PassAtN_16", "Pass@16"), ("PassAtN_64", "Pass@64"))),
    ("table3_majority_sft", "math_sft_majority", "tab:maj_results", (1, 4, 8, 16, 32, 64),
     (("SFT", "SFT baseline"), ("MajVote_N8_Fixed", "Maj@8"), ("MajVote_N16", "Maj@16"), ("MajVote_N64_Frac40", "Maj@64"))),
    ("table4_passn_rl", "math_rl_passn", "tab:rl_sft_comparison", (1, 8, 16, 32),
     (("Standard_RL", "Standard RL"), ("SFT_Weight_N4", "Pass@4"), ("SFT_Weight_N16", "Pass@16"))),
    ("table5_majority_rl", "math_rl_majority", "tab:maj_rl_results", (1, 4, 8, 16),
     (("Standard_RL", "Standard RL"), ("RL_Wt_Maj4", "Maj@4"), ("RL_Wt_Maj8", "Maj@8"))),
    ("table6_unconditional", "protein_unconditional", "tab:bon_scaling", (1, 2, 4, 8, 16, 32, 64),
     (("Standard RL", "Standard RL"), ("BoN-2 (Standard)", "BoN-2"), ("BoN-4 (Standard)", "BoN-4"), ("BoN-8 (Standard)", "BoN-8"))),
    ("table6_conditional", "protein_conditional", "tab:bon_scaling", (1, 2, 4, 8, 16, 32),
     (("Standard RL", "Standard RL"), ("BoN-2 RL", "BoN-2"), ("BoN-4 RL", "BoN-4"), ("BoN-8 RL", "BoN-8"))),
    ("appendix_passn_rl_weights", "math_rl_passn", "tab:rl_full_appendix", (1, 4, 8, 16, 32),
     (("Standard_RL", "Standard RL"), ("SFT_Weight_N4", "Pass@4 (Log)"), ("SFT_Weight_N16", "Pass@16 (Log)"),
      ("RL_Weight_N4", "Pass@4 (Raw)"), ("RL_Weight_N16", "Pass@16 (Raw)"))),
    ("appendix_majority_rl_weights", "math_rl_majority", "tab:maj_ablation_results", (1, 4, 8, 16),
     (("Standard_RL", "Standard RL"), ("SFT_Wt_Maj4", "SFT weight Maj4"), ("SFT_Wt_Maj8", "SFT weight Maj8"),
      ("RL_Wt_Maj4", "RL weight Maj4"), ("RL_Wt_Maj8", "RL weight Maj8"))),
)
