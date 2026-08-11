"""
Are the differences between the models real, or is it noise?

`train.py` reports SVM 99.32%, LR 98.67%, NB 94.22% and the model card prints
them in that order, which invites the reader to treat the ranking as a fact.
On a 7,820-row test set a 0.65-point gap might be a genuine difference or it
might be forty coin flips; nothing in the pipeline currently says which. This
module supplies the missing number:

  - McNemar's test on every pair of systems (paired - the same test articles
    go through both models, so an unpaired test would be the wrong tool)
  - Bootstrap percentile confidence intervals for each system's accuracy and F1
  - Bootstrap CIs for the pairwise *difference* in accuracy, which is the
    interval that actually answers "is A better than B"
  - Holm-Bonferroni adjustment, because six pairwise tests on one test set
    will eventually hand you a p < 0.05 if you let them
  - The comparison this project actually turns on: each trained model against
    the one-rule "mentions Reuters -> real" baseline. If the best model is not
    statistically distinguishable from grepping for a single word, that is the
    finding, and it is worth more than the accuracy table.

The test split is rebuilt from `models/run_metadata.json` exactly the way
`evaluate.py` does it, so the accuracies here match `models/metrics.json`.

Usage:
    python src/significance.py
    python src/significance.py --bootstrap 5000 --exact-max 200
    python src/significance.py --bootstrap 200 --no-figures

Writes:
    reports/significance_report.md
    reports/significance_scores.csv
    reports/significance_pairwise.csv
    reports/figures/significance.png
"""

import argparse
import itertools
import json
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import (FIGURES_DIR, METRICS_JSON, MODEL_KEYS, MODELS_DIR, REAL,
                    REPORTS_DIR, RUN_META, VECTORIZER_FILE, ensure_dirs,
                    model_name, model_path)
from preprocessing import clean_text, load_data

# The one-rule baseline. Same regex and same raw text column that
# train.py's baseline_scores() uses, so the accuracy is directly comparable
# with models/metrics.json -> baselines.one_rule_mentions_reuters.
RULE_KEY = "rule"
RULE_NAME = 'One rule: "mentions Reuters -> real"'
RULE_PATTERN = r"\breuters\b"

COLORS = {"nb": "#e07a5f", "svm": "#3d5a80", "lr": "#81b29a", RULE_KEY: "#6c757d"}
SHORT = {"nb": "NB", "svm": "SVM", "lr": "LR", RULE_KEY: "Reuters rule"}


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test whether the differences between the trained models are "
                    "statistically real (McNemar + bootstrap CIs)."
    )
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Bootstrap resamples for the confidence intervals (0 = skip)")
    p.add_argument("--ci-level", type=float, default=95.0,
                   help="Confidence level for the percentile intervals")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level used for the verdict column")
    p.add_argument("--exact-max", type=int, default=25,
                   help="Use the exact binomial McNemar test when the discordant "
                        "count b+c is at or below this; chi-square above it")
    p.add_argument("--seed", type=int, default=42, help="Seed for the bootstrap resampler")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
    return p.parse_args()


def save_fig(fig, name: str, enabled: bool = True) -> str:
    """Save a figure into reports/figures and return its relative path."""
    if not enabled:
        plt.close(fig)
        return ""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal dataframe -> markdown table (avoids a `tabulate` dependency)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join([out.index.name or ""] + [str(c) for c in out.columns]) + " |"
    sep = "|" + "|".join(["---"] * (len(out.columns) + 1)) + "|"
    rows = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(out.index, out.values)
    ]
    return "\n".join([header, sep] + rows)


def fmt_p(p: float) -> str:
    """p-values span many orders of magnitude here; 0.0000 would be a lie."""
    if pd.isna(p):
        return ""
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def rebuild_split(models_dir: Path, data_dir):
    """
    Recreate train.py's exact split from run_metadata.json so every number
    here lines up with models/metrics.json. Same logic as evaluate.py.
    """
    meta_path = models_dir / RUN_META
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python src/train.py` first - significance.py "
            "needs the run metadata to reproduce the same test split."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df = load_data(
        data_dir,
        strip_boilerplate=meta.get("strip_boilerplate", False),
        drop_duplicates=not meta.get("keep_duplicates", False),
    )
    df["clean_content"] = df["content"].map(clean_text)
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)

    idx_train, idx_test = train_test_split(
        df.index, test_size=meta["test_size"],
        random_state=meta["random_state"], stratify=df["label"],
    )
    return df, idx_train, idx_test, meta


def one_rule_predictions(raw_text: pd.Series) -> np.ndarray:
    """"Mentions Reuters -> real". The whole classifier, in one line."""
    return raw_text.str.contains(RULE_PATTERN, case=False, na=False).astype(int).to_numpy()


# --- the tests -------------------------------------------------------------

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray, exact_max: int = 25) -> dict:
    """
    McNemar's test on two boolean correctness vectors over the *same* items.

    Only the discordant items carry information: b = A right where B is wrong,
    c = A wrong where B is right. Under the null "the two systems are equally
    likely to be the one that gets a discordant item right", b ~ Binomial(b+c, 0.5).

    Both p-values are always computed and reported. `p_raw` is the one that
    feeds the multiple-comparison correction: the exact binomial when b+c is at
    or below `exact_max`, otherwise the chi-square. Reporting both is cheap and
    it lets the reader see how much the normal approximation is costing - deep
    in the tail the two disagree by many orders of magnitude, though at that
    point neither number means anything beyond "not chance".
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_disc = b + c

    if n_disc == 0:
        # The two systems made identical predictions on every test item, so
        # there is nothing to test.
        return {"b": b, "c": c, "n_discordant": 0, "chi2": np.nan,
                "p_exact": 1.0, "p_chi2": np.nan, "p_raw": 1.0,
                "method": "none (identical predictions)"}

    stat = (abs(b - c) - 1) ** 2 / n_disc  # Edwards' continuity correction
    p_exact = float(binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue)
    p_chi2 = float(chi2.sf(stat, df=1))

    use_exact = n_disc <= exact_max
    return {"b": b, "c": c, "n_discordant": n_disc, "chi2": float(stat),
            "p_exact": p_exact, "p_chi2": p_chi2,
            "p_raw": p_exact if use_exact else p_chi2,
            "method": "exact binomial" if use_exact else "chi-square (cc)"}


def holm_bonferroni(pvals) -> np.ndarray:
    """
    Holm-Bonferroni step-down adjusted p-values.

    Six pairwise tests on one test set means six chances to find a "significant"
    difference by luck. Holm controls the family-wise error rate and is
    uniformly more powerful than plain Bonferroni, so there is no reason to
    use the latter.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def bootstrap_draws(y_true: np.ndarray, preds: dict, n_boot: int, rng):
    """
    Paired bootstrap over test items.

    Every system is scored on the *same* resample, so subtracting two accuracy
    columns gives the distribution of the difference with the correlation
    between the systems already accounted for. Resampling each system
    independently would inflate the spread of the difference badly, because
    the systems agree on most items.

    Returns (accuracy_draws, f1_draws), each shaped (n_boot, n_systems),
    with columns in the order of `preds`.
    """
    keys = list(preds)
    n = len(y_true)
    correct = np.column_stack([preds[k] == y_true for k in keys])
    tp = np.column_stack([(preds[k] == REAL) & (y_true == REAL) for k in keys])
    fp = np.column_stack([(preds[k] == REAL) & (y_true != REAL) for k in keys])
    fn = np.column_stack([(preds[k] != REAL) & (y_true == REAL) for k in keys])

    acc = np.empty((n_boot, len(keys)))
    f1 = np.empty((n_boot, len(keys)))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        acc[i] = correct[idx].mean(axis=0)
        t, f_p, f_n = tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0)
        denom = 2 * t + f_p + f_n
        f1[i] = np.where(denom > 0, 2 * t / np.maximum(denom, 1), 0.0)
    return acc, f1


def percentile_ci(draws: np.ndarray, level: float):
    """Two-sided percentile interval, column-wise."""
    tail = (100.0 - level) / 2.0
    return np.percentile(draws, tail, axis=0), np.percentile(draws, 100.0 - tail, axis=0)


# --- figure ----------------------------------------------------------------

def plot_significance(order, acc, acc_ci, pair_rows, level, enabled=True):
    """Left: accuracy with CI, rule as a line. Right: forest plot of differences."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    model_keys = [k for k in order if k != RULE_KEY]
    x = np.arange(len(model_keys))
    point = np.array([acc[k] for k in model_keys])
    lo = np.array([acc_ci[k][0] for k in model_keys])
    hi = np.array([acc_ci[k][1] for k in model_keys])

    axes[0].errorbar(x, point, yerr=[point - lo, hi - point], fmt="o", capsize=5,
                     markersize=7, linewidth=1.6,
                     color="#333333", ecolor="#333333", zorder=3)
    for i, k in enumerate(model_keys):
        axes[0].plot(x[i], point[i], "o", color=COLORS[k], markersize=7, zorder=4)
        axes[0].annotate(f"{point[i]:.4f}", (x[i], hi[i]), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=8)

    if RULE_KEY in acc:
        r_lo, r_hi = acc_ci[RULE_KEY]
        axes[0].axhspan(r_lo, r_hi, color=COLORS[RULE_KEY], alpha=0.18, zorder=1)
        axes[0].axhline(acc[RULE_KEY], color=COLORS[RULE_KEY], linestyle="--",
                        linewidth=1.6, zorder=2,
                        label=f"one-rule baseline ({acc[RULE_KEY]:.4f})")
        axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    axes[0].set_xticks(x, [SHORT[k] for k in model_keys])
    axes[0].set_xlim(-0.6, len(model_keys) - 0.4)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title(f"Accuracy with {level:.0f}% bootstrap CI")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Forest plot: difference in accuracy, A - B, with the paired bootstrap CI.
    y = np.arange(len(pair_rows))
    diffs = np.array([r["acc_diff"] for r in pair_rows])
    d_lo = np.array([r["diff_ci_low"] for r in pair_rows])
    d_hi = np.array([r["diff_ci_high"] for r in pair_rows])
    excludes_zero = (d_lo > 0) | (d_hi < 0)

    for i in range(len(pair_rows)):
        color = "#3d5a80" if excludes_zero[i] else "#adb5bd"
        axes[1].plot([d_lo[i], d_hi[i]], [y[i], y[i]], color=color, linewidth=2.2)
        axes[1].plot(diffs[i], y[i], "o", color=color, markersize=6)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y, [r["label"] for r in pair_rows], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("difference in accuracy (A - B)")
    axes[1].set_title(f"Pairwise difference, {level:.0f}% CI\n"
                      "(grey = interval contains 0)", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    return save_fig(fig, "significance.png", enabled)


# --- report ----------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dirs()
    # The figure is a plot of confidence intervals, so it needs the bootstrap.
    figs = not args.no_figures and args.bootstrap > 0
    models_dir = Path(args.models_dir)
    rng = np.random.default_rng(args.seed)

    print("[1/6] Rebuilding the training run's test split...")
    df, _, idx_test, meta = rebuild_split(models_dir, args.data_dir)
    y_test = df.loc[idx_test, "label"].to_numpy()
    raw_test = df.loc[idx_test, "content"]
    n_test = len(y_test)
    print(f"    {n_test:,} test articles "
          f"(boilerplate stripped: {meta.get('strip_boilerplate')})")

    print("[2/6] Scoring models and the one-rule baseline...")
    vectorizer = joblib.load(models_dir / VECTORIZER_FILE)
    X_test = vectorizer.transform(df.loc[idx_test, "clean_content"])

    preds, names = {}, {}
    for key in MODEL_KEYS:
        path = model_path(key, models_dir)
        if not path.exists():
            print(f"    skipping {model_name(key)} ({path.name} not found)")
            continue
        preds[key] = joblib.load(path).predict(X_test)
        names[key] = model_name(key)
    if not preds:
        raise SystemExit("No trained models found - run `python src/train.py` first.")

    preds[RULE_KEY] = one_rule_predictions(raw_test)
    names[RULE_KEY] = RULE_NAME

    correct = {k: (v == y_test) for k, v in preds.items()}
    acc = {k: float(v.mean()) for k, v in correct.items()}
    f1 = {k: float(f1_score(y_test, v, pos_label=REAL)) for k, v in preds.items()}
    for k in preds:
        print(f"    {names[k]:<34} acc={acc[k]:.4f}  f1={f1[k]:.4f}")

    # Reproducibility check: these should be the exact numbers train.py stored.
    metrics_note = "`models/metrics.json` not found, so no cross-check was possible."
    metrics_path = models_dir / METRICS_JSON
    if metrics_path.exists():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        drift = {}
        for k in preds:
            ref = (stored.get("baselines", {}).get("one_rule_mentions_reuters")
                   if k == RULE_KEY else stored.get("models", {}).get(k, {}).get("accuracy"))
            if ref is not None:
                drift[names[k]] = abs(ref - acc[k])
        worst = max(drift.values()) if drift else 0.0
        metrics_note = (
            "Every accuracy below reproduces `models/metrics.json` "
            + ("exactly" if worst == 0.0 else f"to within {worst:.2e}")
            + ", so this is the same test split the models were scored on."
            if worst < 1e-9 else
            f"**Warning:** accuracies differ from `models/metrics.json` by up to "
            f"{worst:.2e}. The split or the data has changed since training; "
            f"re-run `python src/train.py` before quoting anything here."
        )
        print(f"    cross-check vs metrics.json: max drift {worst:.2e}")

    # Rank by accuracy so that "A vs B" mostly reads as "better vs worse".
    order = sorted(preds, key=lambda k: acc[k], reverse=True)

    print(f"[3/6] Paired bootstrap ({args.bootstrap:,} resamples)...")
    if args.bootstrap > 0:
        acc_draws, f1_draws = bootstrap_draws(y_test, {k: preds[k] for k in order},
                                              args.bootstrap, rng)
        a_lo, a_hi = percentile_ci(acc_draws, args.ci_level)
        f_lo, f_hi = percentile_ci(f1_draws, args.ci_level)
        acc_ci = {k: (float(a_lo[i]), float(a_hi[i])) for i, k in enumerate(order)}
        f1_ci = {k: (float(f_lo[i]), float(f_hi[i])) for i, k in enumerate(order)}
    else:
        acc_draws = None
        acc_ci = {k: (np.nan, np.nan) for k in order}
        f1_ci = {k: (np.nan, np.nan) for k in order}
        print("    skipped (--bootstrap 0): no confidence intervals in this run")

    scores = pd.DataFrame({
        names[k]: {
            "accuracy": acc[k],
            "acc_ci_low": acc_ci[k][0],
            "acc_ci_high": acc_ci[k][1],
            "f1": f1[k],
            "f1_ci_low": f1_ci[k][0],
            "f1_ci_high": f1_ci[k][1],
            "errors": int((~correct[k]).sum()),
        } for k in order
    }).T
    scores["errors"] = scores["errors"].astype(int)
    scores.index.name = "system"

    print("[4/6] McNemar tests + Holm-Bonferroni...")
    col_of = {k: i for i, k in enumerate(order)}
    rows = []
    for a, b in itertools.combinations(order, 2):
        res = mcnemar(correct[a], correct[b], args.exact_max)
        diff = acc[a] - acc[b]
        if acc_draws is not None:
            d = acc_draws[:, col_of[a]] - acc_draws[:, col_of[b]]
            d_lo, d_hi = percentile_ci(d[:, None], args.ci_level)
            d_lo, d_hi = float(d_lo[0]), float(d_hi[0])
        else:
            d_lo = d_hi = np.nan
        rows.append({
            "A": a, "B": b,
            "label": f"{SHORT[a]} vs {SHORT[b]}",
            "comparison": f"{names[a]} vs {names[b]}",
            "acc_diff": diff, "diff_ci_low": d_lo, "diff_ci_high": d_hi,
            **res,
        })
        print(f"    {SHORT[a]:>12} vs {SHORT[b]:<12} b={res['b']:<5} c={res['c']:<5} "
              f"p={res['p_raw']:.3g} ({res['method']})")

    p_adj = holm_bonferroni([r["p_raw"] for r in rows])
    for r, p in zip(rows, p_adj):
        r["p_holm"] = float(p)
        if p < args.alpha:
            r["verdict"] = f"{SHORT[r['A']]} better" if r["acc_diff"] > 0 else f"{SHORT[r['B']]} better"
        else:
            r["verdict"] = "no difference detected"

    pair_df = pd.DataFrame(rows)
    pair_out = pair_df[["comparison", "b", "c", "n_discordant", "chi2", "p_exact",
                        "p_chi2", "p_raw", "p_holm", "method", "acc_diff",
                        "diff_ci_low", "diff_ci_high", "verdict"]].copy()
    pair_out.to_csv(REPORTS_DIR / "significance_pairwise.csv", index=False)
    scores.to_csv(REPORTS_DIR / "significance_scores.csv")

    # Markdown version of the pairwise table. Short labels, because the full
    # model names make the table too wide to read; the CSV keeps them.
    pretty = pd.DataFrame({
        "b": [f"{r['b']:,}" for r in rows],
        "c": [f"{r['c']:,}" for r in rows],
        "b+c": [f"{r['n_discordant']:,}" for r in rows],
        "chi2": ["" if pd.isna(r["chi2"]) else f"{r['chi2']:.2f}" for r in rows],
        "p_exact": [fmt_p(r["p_exact"]) for r in rows],
        "p_chi2": [fmt_p(r["p_chi2"]) for r in rows],
        "used": [r["method"] for r in rows],
        "p_holm": [fmt_p(r["p_holm"]) for r in rows],
        "acc_diff": [f"{r['acc_diff']:+.4f}" for r in rows],
        f"{args.ci_level:.0f}% CI on diff": [
            f"[{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}]"
            if pd.notna(r["diff_ci_low"]) else "" for r in rows],
        "verdict": [r["verdict"] for r in rows],
    }, index=[r["label"] for r in rows])
    pretty.index.name = "A vs B"

    # Square matrix of Holm-adjusted p-values, which is what "pairwise matrix"
    # usually means in a report.
    matrix = pd.DataFrame("-", index=[names[k] for k in order],
                          columns=[SHORT[k] for k in order])
    for r in rows:
        matrix.loc[names[r["A"]], SHORT[r["B"]]] = fmt_p(r["p_holm"])
        matrix.loc[names[r["B"]], SHORT[r["A"]]] = fmt_p(r["p_holm"])
    matrix.index.name = "system"

    # How much of each model is just the rule in disguise?
    rule_pred = preds[RULE_KEY]
    agree = pd.DataFrame({
        names[k]: {
            "agreement_with_rule": float((preds[k] == rule_pred).mean()),
            "both_right": float((correct[k] & correct[RULE_KEY]).mean()),
            "model_right_rule_wrong": float((correct[k] & ~correct[RULE_KEY]).mean()),
            "rule_right_model_wrong": float((~correct[k] & correct[RULE_KEY]).mean()),
            "both_wrong": float((~correct[k] & ~correct[RULE_KEY]).mean()),
        } for k in order if k != RULE_KEY
    }).T
    agree.index.name = "model"
    top_agree = agree["agreement_with_rule"].idxmax()
    agreement_note = (
        f"{top_agree} makes the same call as the keyword rule on "
        f"{agree.loc[top_agree, 'agreement_with_rule']:.1%} of test articles, and both "
        f"are wrong together on {agree.loc[top_agree, 'both_wrong']:.2%}. Whatever the "
        f"model learned, it is close to a re-implementation of the rule; the "
        f"disagreements are too few to establish that it learned anything else."
    )

    print("[5/6] Figure...")
    fig_path = plot_significance(order, acc, acc_ci, rows, args.ci_level, figs)

    print("[6/6] Writing report...")
    best_model = max((k for k in preds if k != RULE_KEY), key=lambda k: acc[k])
    vs_rule = next(r for r in rows
                   if {r["A"], r["B"]} == {best_model, RULE_KEY})
    rule_beats_best = acc[RULE_KEY] > acc[best_model]
    rule_significant = vs_rule["p_holm"] < args.alpha

    # McNemar's b/c are reported A-then-B; state them model-first regardless of
    # which way round the pair was built.
    if vs_rule["A"] == best_model:
        model_only, rule_only = vs_rule["b"], vs_rule["c"]
    else:
        model_only, rule_only = vs_rule["c"], vs_rule["b"]
    disagree = (
        f"The two disagree on only {vs_rule['n_discordant']} of {n_test:,} test "
        f"articles, and split those near-evenly: {model_only} the model gets right "
        f"and the rule misses, {rule_only} the other way round."
    )

    if rule_significant and rule_beats_best:
        headline = (
            f"The one-rule baseline is **significantly more accurate** than "
            f"{names[best_model]}, the best trained model "
            f"({acc[RULE_KEY]:.4f} vs {acc[best_model]:.4f}, Holm-adjusted "
            f"p = {fmt_p(vs_rule['p_holm'])}). {disagree}"
        )
    elif rule_significant:
        headline = (
            f"{names[best_model]} is significantly more accurate than the one-rule "
            f"baseline (Holm-adjusted p = {fmt_p(vs_rule['p_holm'])}), but the "
            f"margin is {abs(vs_rule['acc_diff']):.4f} accuracy - "
            f"{abs(vs_rule['acc_diff']) * n_test:.0f} articles out of {n_test:,}. "
            f"{disagree}"
        )
    else:
        ci_clause = (
            f", {args.ci_level:.0f}% CI on the difference "
            f"[{vs_rule['diff_ci_low']:+.4f}, {vs_rule['diff_ci_high']:+.4f}], "
            f"which contains zero" if pd.notna(vs_rule["diff_ci_low"]) else ""
        )
        headline = (
            f"{names[best_model]}, the best trained model, is **not statistically "
            f"distinguishable** from a one-line rule that greps for the word "
            f"\"Reuters\" ({acc[best_model]:.4f} vs {acc[RULE_KEY]:.4f}; "
            f"Holm-adjusted p = {fmt_p(vs_rule['p_holm'])}{ci_clause}). "
            f"{disagree} On this test set the "
            f"{meta.get('n_features', 0):,}-feature TF-IDF pipeline buys nothing "
            f"measurable over the keyword."
        )

    n_sig = sum(1 for r in rows if r["p_holm"] < args.alpha)
    n_flipped = sum(1 for r in rows if r["p_raw"] < args.alpha <= r["p_holm"])
    n_exact = sum(1 for r in rows if r["method"].startswith("exact"))

    # The smallest gap the test can still call significant, and the biggest one
    # it cannot - together these say what this test set can and cannot resolve.
    sig = sorted((r for r in rows if r["p_holm"] < args.alpha),
                 key=lambda r: abs(r["acc_diff"]))
    nonsig = sorted((r for r in rows if r["p_holm"] >= args.alpha),
                    key=lambda r: abs(r["acc_diff"]), reverse=True)
    resolution = ""
    if sig:
        d = abs(sig[0]["acc_diff"])
        resolution += (
            f"The smallest gap that still comes out significant is "
            f"{sig[0]['label']} at {d:.4f} accuracy - {d * n_test:.0f} articles "
            f"out of {n_test:,}. ")
    if nonsig:
        d = abs(nonsig[0]["acc_diff"])
        resolution += (
            f"The largest gap it cannot resolve is {nonsig[0]['label']} at "
            f"{d:.4f}. ")
    boilerplate_note = (
        "\n> This run's models were trained on **boilerplate-stripped** text, so the\n"
        "> `(Reuters)` tags the one-rule baseline looks for have been removed from\n"
        "> `content` as well. The baseline therefore predicts `fake` almost everywhere\n"
        "> and collapses to roughly the majority-class rate. Compare it against a\n"
        "> non-stripped run before reading anything into that number.\n"
        if meta.get("strip_boilerplate") else ""
    )

    def embed(path, caption):
        return f"\n![{caption}]({path})\n" if path else ""

    report = f"""# Statistical Significance of Model Differences

Generated by `python src/significance.py`. Test split reproduced from
`models/run_metadata.json`: **{n_test:,} articles**,
`random_state={meta['random_state']}`, `test_size={meta['test_size']}`.

{metrics_note}
{boilerplate_note}
## Headline

{textwrap.fill(headline, 88)}

## 1. Why McNemar and not a two-proportion test

Every system here is scored on **the same {n_test:,} test articles**. That makes the
comparisons paired, and the pairing carries most of the information: the models
agree on the overwhelming majority of articles, so the interesting quantity is
not "how often is each right" but "on the articles where they disagree, who is
right more often".

An unpaired two-proportion z-test throws that structure away. It treats the two
accuracy figures as if they came from two independent samples and computes a
standard error that is far too large, which makes it badly under-powered here -
it would call almost every difference on this test set insignificant regardless
of the evidence. McNemar's test conditions on the discordant items only:

- `b` = A correct, B wrong
- `c` = A wrong, B correct

and tests whether `b` and `c` are draws from a fair coin. When `b + c` is small
(at or below `--exact-max`, {args.exact_max} in this run) the exact binomial version is used,
because the chi-square approximation is unreliable in that regime; otherwise the
continuity-corrected chi-square statistic is used. Both p-values are printed in
the table below and the `used` column says which one was carried into the
correction - {n_exact} exact, {len(rows) - n_exact} chi-square in this run.
Near the decision boundary the two agree closely; far into the tail they
diverge by orders of magnitude, which changes no verdict and is worth knowing
before quoting a p-value like 1e-80 as if it were a measurement.

The bootstrap follows the same principle: each resample of test items is scored
by **all** systems, so the interval on the difference reflects how correlated
the systems are. Resampling the systems independently would inflate that
interval and hide real differences.

## 2. Accuracy and F1 with bootstrap confidence intervals

{f"{args.bootstrap:,} resamples, {args.ci_level:.0f}% percentile intervals, seed {args.seed}."
 if args.bootstrap else
 "Run with `--bootstrap 0`, so the interval columns are empty and the figure was skipped."}
F1 is for the positive class (`real` = 1).

{md_table(scores)}
{embed(fig_path, "significance")}

## 3. Pairwise matrix (Holm-adjusted p-values)

Cell = the Holm-adjusted McNemar p-value for the row system against the column
system. Values below {args.alpha} are differences that survive the
multiple-comparison correction.

{md_table(matrix)}

## 4. Pairwise tests in full

{md_table(pretty)}

`b` counts articles the first system got right and the second got wrong; `c` is
the reverse - only these two cells enter the test. `chi2` is the
continuity-corrected statistic, `p_holm` adjusts the p-value named in `used`,
`acc_diff` is A minus B, and the CI is the paired bootstrap interval on that
difference.

{n_sig} of {len(rows)} comparisons are significant after Holm-Bonferroni at
alpha = {args.alpha}.{f" {n_flipped} comparison(s) were significant on the raw p-value but not after correction."
 if n_flipped else " No comparison changed verdict because of the correction."}

## 5. How much of each model is the one-rule baseline?

Share of the test set in each agreement cell, model against
"mentions Reuters -> real". This is not a significance test - it is the
descriptive version of the same point, and it is the reason the p-values above
should not be read as a quality ranking.

{md_table(agree)}

{textwrap.fill(agreement_note, 88)}

## 6. What this does and does not tell you

**It does say:** {n_sig} of the {len(rows)} pairwise comparisons survive Holm
correction, so those gaps are unlikely to be sampling noise. The other
{len(rows) - n_sig} {"is" if len(rows) - n_sig == 1 else "are"} unresolved at this
sample size, which is not the same thing as proven equal.
{textwrap.fill(resolution, 88)}
Statistical detectability and practical importance are different claims, and a
p-value only supports the first.

**It does not say** that the better model is a better fake-news detector. All of
these systems are being measured on a test set drawn from the same leaky
distribution as their training data. `reports/eda_report.md` shows that
`(Reuters)` appears in 99.2% of real articles and essentially no fake ones, and
that all 7 subject categories are exclusive to one class. A statistically
significant win on this data is a statistically significant win at exploiting
that leak. The ranking is real; what it ranks is not what the project claims to
measure.

**Three further caveats:**

1. **One split, fixed models.** Both tests treat the trained models as fixed and
   resample only the test items. Variation from the training split, the seed and
   the vectorizer is not in these intervals. `evaluate.py --cv 5` covers part of
   that gap; the cross-validated F1 standard deviations in `models/metrics.json`
   are the other half of the picture.
2. **Bootstrap CIs are percentile intervals**, not bias-corrected. At these
   sample sizes and accuracies the difference is negligible, but the intervals
   are slightly optimistic near the 1.0 ceiling, where the sampling
   distribution is squashed against the boundary.
3. **The p-values do not transfer.** Run `python src/cross_dataset_eval.py` and
   repeat this analysis there if you want a significance claim that says
   anything about out-of-distribution behaviour.

## 7. Reproduce

```
python src/significance.py --bootstrap {args.bootstrap} --seed {args.seed}
```

Raw tables: `reports/significance_pairwise.csv`, `reports/significance_scores.csv`.
"""

    out = REPORTS_DIR / "significance_report.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"      {REPORTS_DIR / 'significance_pairwise.csv'}")
    print(f"      {REPORTS_DIR / 'significance_scores.csv'}")
    if fig_path:
        print(f"      {FIGURES_DIR / 'significance.png'}")
    print("\n" + textwrap.fill(headline.replace("**", ""), 88))


if __name__ == "__main__":
    main()
