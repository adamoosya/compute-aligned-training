"""Generate paper-facing assets from recovered data, never from invented samples."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .reference import TABLES, curves, diagnostic_rows, load_sources, read_json, reference_audit, sha256


def _write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def _write_csv(path, names, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(names)
        writer.writerows(rows)


def _table(out, stem, names, rows, *, places=2, note):
    _write_csv(out / (stem + ".csv"), names, rows)
    # These are means-only tables. Do not create SEs from aggregate curves.
    text = ["% " + note, r"\begin{tabular}{l" + "r" * (len(names) - 1) + "}", r"\toprule",
            " & ".join(n.replace("%", r"\%") for n in names) + r" \\", r"\midrule"]
    shown = [[f"{float(v):.{places}f}" for v in row[1:]] for row in rows]
    maxima = [max(float(row[j]) for row in shown) for j in range(len(names) - 1)]
    for row, numbers in zip(rows, shown):
        cells = [r"\textbf{" + n + "}" if float(n) == maxima[j] else n for j, n in enumerate(numbers)]
        text.append(str(row[0]).replace("_", r"\_").replace("%", r"\%") + " & " + " & ".join(cells) + r" \\")
    text += [r"\bottomrule", r"\end{tabular}", ""]
    (out / (stem + ".tex")).write_text("\n".join(text), encoding="utf-8")


def _figure(title, xlabel, ylabel):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(7.6, 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=.2)
    return fig, ax


def _save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        dest = directory / (stem + "." + suffix)
        if dest.exists():
            raise FileExistsError(dest)
        fig.savefig(dest, dpi=160, bbox_inches="tight")
    fig.clear()


def _line_plot(directory, stem, title, families, selected, *, ylabel, delta=None, scale=1,
               log_x=False):
    fig, ax = _figure(title, "Inference budget k", ylabel)
    if delta is not None:
        ax.axhline(0, linestyle="--", linewidth=1, label="Baseline")
    for name, label in selected:
        curve = families[name]
        x = list(curve)
        y = [scale * (curve[k] - (families[delta][k] if delta else 0)) for k in x]
        ax.plot(x, y, linewidth=1.7, label=label, marker="o" if len(x) < 10 else None,
                markersize=3)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    _save(fig, directory, stem)


def _histogram(directory, stem, rows, strategy):
    x = np.asarray([row["rho"] for row in rows], dtype=float)
    fig, ax = _figure("Empirical Distribution of Optimization Efficiency\n(" + strategy + ", $N=8$)",
                      r"Optimization Efficiency ($\rho$)", "Frequency (Number of Problems)")
    _, edges, _ = ax.hist(x, bins=15, alpha=.65)
    if len(x) > 1 and x.std(ddof=1) > 0:
        width = x.std(ddof=1) * len(x) ** (-.2)
        grid = np.linspace(min(1.0, x.min()), x.max() + width, 300)
        density = np.exp(-.5 * ((grid[:, None] - x) / width)**2).mean(axis=1) / (width * math.sqrt(2*math.pi))
        ax.plot(grid, density * len(x) * (edges[1] - edges[0]), linewidth=1.8)
    ax.axvline(1, linestyle="--", linewidth=2, label=r"Theoretical Bound ($\rho \geq 1$)")
    ax.axvline(float(x.mean()), linewidth=2, label=rf"Empirical Mean ($\rho = {x.mean():.2f}$)")
    ax.set_xlim(.96, x.max() + .07 * max(1, x.max()-1))
    ax.legend(fontsize=9)
    _save(fig, directory, stem)


def _analytic_assets(directory, data_dir):
    import torch
    from cat.objectives import passn_sft_weight, passn_rl_weight, majority_sft_weight, majority_rl_weight, best_of_n_rl_weight
    p = torch.linspace(0, 1, 1001, dtype=torch.float64)
    for mode, fn, title in [
        ("sft_pass_n", passn_sft_weight, "SFT Scaling Factor for Pass@N"),
        ("rl_pass_n", passn_rl_weight, "RL Scaling Factor for Pass@N"),
        ("sft_maj_n", majority_sft_weight, "SFT Scaling Factor for Majority Vote"),
        ("rl_maj_n", majority_rl_weight, "RL Scaling Factor for Majority Vote"),
        ("rl_bon", best_of_n_rl_weight, "RL Scaling Factor for Best-of-N")]:
        fig, ax = _figure(title, "Base probability p" if "bon" not in mode else "Lower-tail reward quantile", "Scaling factor")
        for n in [1, 4, 16, 64]:
            w = fn(p, n, n//2+1) if "maj" in mode else fn(p, n)
            ax.plot(p.numpy(), w.numpy(), label=f"N={n}")
        ax.legend(); _save(fig, directory, "plot_" + mode + "_raw")
    fig, ax = _figure("RL Majority Vote: varying thresholds (N=64)", "Base probability p", "Scaling factor")
    for k in [6, 16, 24, 33]:
        ax.plot(p.numpy(), majority_rl_weight(p, 64, k).numpy(), label=f"k={k}")
    ax.legend(); _save(fig, directory, "plot_rl_maj_k_raw")
    # The support-comparison panels use SFT weights on both sides, not an RL derivative labelled SFT.
    for strategy, w in [("passn", passn_sft_weight(p, 16)), ("majority", majority_sft_weight(p, 16, 4))]:
        fig, ax = _figure(f"SFT weight comparison: {strategy}", "Base probability p", "SFT scaling factor")
        ax.plot(p.numpy(), w.numpy(), label="CAT")
        ax.axhline(1, linestyle="--", label="Standard SFT")
        ax.legend(); _save(fig, directory, "gradient_alignment_" + strategy)
    # A uniform-grid cosine of scalar weight functions, not a guarantee about model gradients.
    grid = torch.linspace(0, 1, 2001, dtype=torch.float64)
    ws = {n: passn_sft_weight(grid, n).numpy() for n in [1, 4, 16, 64]}
    # np.trapezoid is unavailable on NumPy 1.26; use an explicit trapezoidal sum.
    def integral(y):
        return float(np.sum((y[:-1] + y[1:]) * .5) / (len(y)-1))
    rows = []
    for a, wa in ws.items():
        for b, wb in ws.items():
            rows.append([a, b, integral(wa*wb) / math.sqrt(integral(wa*wa)*integral(wb*wb))])
    _write_csv(data_dir / "passn_weight_alignment.csv", ["train_n", "test_n", "uniform_weight_cosine"], rows)


def build_assets(reference_root, output, *, plots=True):
    reference_root, output = Path(reference_root), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new output directory: {output}")
    manifest, sources = load_sources(reference_root)
    data = curves(sources)
    audit = reference_audit(reference_root)
    reported = read_json(reference_root / "reported_appendix.json")
    if plots:
        import matplotlib  # fail before writing when the plots extra is missing
    output.mkdir(parents=True, exist_ok=False)
    table_dir, fig_dir, data_dir = (output / name for name in ("tables", "figures", "data"))
    for directory in (table_dir, fig_dir, data_dir): directory.mkdir()
    _write_json(output / "reference_audit.json", audit)
    for stem, family, label, budgets, models in TABLES:
        percent = family.startswith("math")
        multiplier = 100 if percent else 1
        rows = [[pretty, *[multiplier * data[family][saved][k] for k in budgets]] for saved, pretty in models]
        names = ["Model", *[f"@{k}" + (" (%)" if percent else "") for k in budgets]]
        _table(table_dir, stem, names, rows, places=1 if percent and not stem.startswith("appendix") else 2,
               note=f"Recovered means for {label}. SEs not recomputed; rounding once from saved values.")
    for table in reported["tables"]:
        _table(table_dir, table["id"], ["Model", *table["columns"]],
               [[k, *v] for k, v in table["rows"].items()],
               note="MANUSCRIPT TRANSCRIPTION ONLY: original numerical logs were not recovered.")
    (table_dir / "table1_weights.tex").write_text(
        r"\begin{tabular}{lll}" + "\n" + r"\toprule Strategy & SFT & RL \\" + "\n" +
        r"Pass@$N$ & $\frac{Np(1-p)^{N-1}}{1-(1-p)^N}$ & $N(1-p)^{N-1}$ \\" + "\n" +
        r"Majority Vote & $\frac{k\binom{N}{k}p^k(1-p)^{N-k}}{\sum_{i=k}^N\binom{N}{i}p^i(1-p)^{N-i}}$ & $N\binom{N-1}{k-1}p^{k-1}(1-p)^{N-k}$ \\" + "\n" +
        r"Best-of-$N$ & $\frac{Np(1-p)^{N-1}}{1-(1-p)^N}$ & $N(P_{<y})^{N-1}$ \\" + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")
    for key in ("majority_diagonal", "bon_diagonal"):
        rows = diagnostic_rows(sources, key)
        _write_csv(data_dir / (key + ".csv"), ["p_val", "rho_original", "rho"],
                   [[r[k] for k in ("p_val", "rho_original", "rho")] for r in rows])
        if plots:
            _histogram(fig_dir, "diagonal_approximation_histogram_corrected" if key.startswith("majority") else "bon_diagonal_histogram",
                       rows, "Majority Vote" if key.startswith("majority") else "Best-of-N")
    if plots:
        # Main and ablation curves: connect only the saved budgets, without smoothing or interpolation.
        for i, (stem, family, label, budgets, models) in enumerate(TABLES):
            is_math = family.startswith("math")
            filename = {0:"pass_at_k_final",1:"final_champions_maj_scaling",2:"main_rl_sft_deltas",3:"main_figure_linear",4:"scaling_plot_clean",5:"bon_scaling_comparison",6:"appendix_full_scaling",7:"appendix_figure_linear"}[i]
            title = {0:"Pass@k Improvement over SFT",1:"Maj@k Improvement over SFT\n(Champion Models)",2:"Performance Gain over Standard RL",3:"Performance Gain: Strategy-Aligned RL vs Standard RL",4:"Scaling Laws: Unconditional Hydrophobicity",5:"Inference Scaling: Standard vs BoN",6:"Pass@k: Raw and Log-Weighted RL",7:"Majority Vote: SFT and RL Weights"}[i]
            delta = models[0][0] if i < 4 else None
            selected = models[1:] if delta else models
            _line_plot(fig_dir, filename, title, data[family], selected, delta=delta,
                       scale=100 if is_math else 1,
                       ylabel="Improvement (percentage points)" if delta else ("Accuracy (%)" if is_math else "Expected max reward"),
                       log_x=not is_math)
        sweep = [("MajVote_N64_Frac25","Fraction 0.25 (k=16)"),("MajVote_N64_Frac33","Fraction 0.33 (k=22)"),("MajVote_N64_Frac40","Fraction 0.40 (k=26)")]
        _line_plot(fig_dir,"Maj64_Sweep","Maj64 Hyperparameter Sweep (Delta over SFT)",data["math_sft_majority"],sweep,
                   delta="SFT",scale=100,ylabel="Improvement (percentage points)")
        fig, ax = _figure("Hydrophobicity Distribution", "Hydrophobicity ratio", "Density")
        for name, row in sources["protein_unconditional"].items():
            values = np.asarray(row["ratios"], dtype=float)
            if values.size < 2 or not np.isfinite(values).all():raise ValueError("Invalid hydrophobicity values")
            ax.hist(values, bins=np.linspace(0,1,26), density=True, histtype="step", linewidth=1.6, label=name)
        ax.legend(fontsize=9); _save(fig, fig_dir, "density_plot_clean")
        for index, (name,row) in enumerate(sources["protein_conditional"].items()):
            x, y = np.asarray(row["scatter_x"]), np.asarray(row["scatter_y"])
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Invalid conditional scatter data")
            fig, ax = _figure(name, "Input hydrophobicity", "Output hydrophobicity")
            ax.scatter(x, y, s=14, alpha=.5, label="Saved first candidate per prompt")
            ax.plot([0,1],[1,0],linestyle="--",label="Target: 1 - input")
            ax.set(xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
            _save(fig, fig_dir, f"bon_conditional_panel_{index+1}")
        _analytic_assets(fig_dir, data_dir)
    notes = """# Generated assets

The main and weighting-ablation tables contain **recovered means only**. No
standard errors were inferred from aggregate means. Two paper display values in
Table 4 differ because these files round once from the original saved numbers;
see reference_audit.json. Raw inputs have not been altered.

`token_margin_reported` and `warmup_rl_reported` are clearly marked manuscript
transcriptions; their original logs are unavailable. They are not new experiments.

Figure 9 applies the reviewed sign correction to the original 81 saved ratios,
once. Figure 10 uses all 277 ratios as saved. Neither is a new sampling run.

Plots are redraws, not pixel-identical copies of the paper. Conditional protein
panels are separate files; only the saved first-candidate scatter values are
available. Curves use saved budgets only. Analytical weight plots use the
implemented formulas, not a model. The uniform weight cosine is a descriptive
function comparison, not a validated model-performance predictor.
"""
    (output / "README.md").write_text(notes, encoding="utf-8")
    products = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    _write_json(output / "complete.json", {"kind":"paper_asset_redraw_from_recovered_results","sources":manifest["sources"],
                                           "files":products,"plots":plots,"standard_errors_recomputed":False})
    return {"directory":str(output), "files":len(products), "figures":len(list(fig_dir.glob("*.png"))),
            "tables":len(list(table_dir.glob("*.tex"))), "audit":audit}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=Path("results/reference"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args(argv)
    try:
        r=build_assets(args.reference,args.output,plots=not args.no_plots)
    except (ValueError,OSError,ImportError,KeyError) as e:
        parser.exit(2,f"error: {e}\n")
    print(f"Historical sources verified: {r['audit']['sources_verified']}")
    print(f"Paper display matches: {r['audit']['display_matches']}/{r['audit']['display_comparisons']} (two recorded rounding differences)")
    for key,d in r['audit']['diagnostics'].items():
        print(f"{key}: {d['count']} rows; mean {d['mean']:.6f}")
    print(f"Wrote {r['tables']} table files and {r['figures']} figures to {r['directory']}")
    print("Paper assets generated from saved results; no model runs.")
    return 0
