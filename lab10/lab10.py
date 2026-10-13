"""Lab 10: score calibration, score-level fusion, and final evaluation."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

LABS_FOLDER = Path(__file__).resolve().parents[1]
for folder in ["lab02", "lab06", "lab07", "lab08", "lab09"]:
    sys.path.insert(0, str(LABS_FOLDER / folder))
from lab02 import load_fingerprints
from lab06 import evaluate_scores, bayes_error_curve
from lab07 import train_binary, binary_scores, quadratic_features
from lab08 import rbf_kernel, kernel_scores
from lab09 import unpack_gmm, logpdf_GMM

NAMES = ["GMM", "Logistic regression", "SVM", "Fusion"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# 1. Calibration is unregularized, prior-weighted logistic regression on scores.

def score_matrix(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if scores.ndim != 2 or min(scores.shape) == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite system-by-sample matrix or a vector.")
    return scores


def fit_calibration(scores, labels, prior):
    model = train_binary(score_matrix(scores), labels, 0, target_prior=prior)
    # Store the LLR intercept directly, so the prior is subtracted exactly once.
    return {"w": model["w"], "offset": model["b"] - np.log(prior / (1 - prior)),
            "prior": float(prior)}


def apply_calibration(model, scores):
    scores = score_matrix(scores)
    if scores.shape[0] != model["w"].size:
        raise ValueError("The calibration model and scores have different system counts.")
    return model["w"] @ scores + model["offset"]


def kfold_calibration(scores, labels, prior, folds=5):
    """Return held-out calibrated scores in the ORIGINAL sample order."""
    scores = score_matrix(scores)
    labels = np.asarray(labels)
    if labels.shape != (scores.shape[1],) or not isinstance(folds, int) or not 2 <= folds <= labels.size:
        raise ValueError("Use matching labels and between two and N folds.")
    indices = np.arange(labels.size)
    calibrated = np.empty(labels.size)
    models = []
    for fold in range(folds):
        held_out = indices[fold::folds]
        training = np.hstack([indices[other::folds] for other in range(folds) if other != fold])
        model = fit_calibration(scores[:, training], labels[training], prior)
        calibrated[held_out] = apply_calibration(model, scores[:, held_out])
        models.append(model)
    return calibrated, models


def cost(scores, labels, prior):
    return evaluate_scores(scores, labels, [(prior, 1, 1)])[0]


def report(title, scores, labels, names, prior):
    print(f"\n{title} — prior {prior:g}, unit costs")
    print(f"{'System':<29} {'minDCF':>10} {'actDCF':>10}")
    results = [cost(row, labels, prior) for row in scores]
    for name, result in zip(names, results):
        print(f"{name:<29} {result['minimum']:10.6f} {result['actual']:10.6f}")
    return results


# 2. The supplied two-system exercise: single-fold and five-fold approaches.

def load_example(folder, prefix=""):
    scores = np.vstack([np.load(folder / f"{prefix}scores_{index}.npy", allow_pickle=False)
                        for index in [1, 2]])
    labels = np.load(folder / f"{prefix}labels.npy", allow_pickle=False)
    if labels.shape != (scores.shape[1],) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Example scores need matching binary labels containing both classes.")
    return score_matrix(scores), labels


def run_examples(folder):
    scores, labels = load_example(folder)
    prior = 0.2
    report("Original full development set", scores, labels, ["System 1", "System 2"], prior)
    training = np.arange(labels.size)[::3]
    validation = np.hstack([np.arange(labels.size)[1::3], np.arange(labels.size)[2::3]])
    inputs = [scores[0:1], scores[1:2], scores]
    single_models = [fit_calibration(rows[:, training], labels[training], prior) for rows in inputs]
    single_scores = np.vstack([apply_calibration(model, rows[:, validation])
                               for model, rows in zip(single_models, inputs)])
    pooled_scores = np.vstack([kfold_calibration(rows, labels, prior)[0] for rows in inputs])
    # Final five-fold approach: fit one new mapping using ALL development scores.
    final_models = [fit_calibration(rows, labels, prior) for rows in inputs]
    evaluation, evaluation_labels = load_example(folder, "eval_")
    evaluation_inputs = [evaluation[0:1], evaluation[1:2], evaluation]
    names = ["System 1 calibrated", "System 2 calibrated", "Fusion"]
    figures = []
    for title, raw, calibrated, local_labels, models in [
            ("Single-fold", scores[:, validation], single_scores, labels[validation], single_models),
            ("Five-fold", scores, pooled_scores, labels, final_models)]:
        eval_calibrated = np.vstack([apply_calibration(model, rows)
                                    for model, rows in zip(models, evaluation_inputs)])
        report(title + " validation: raw", raw, local_labels, ["System 1", "System 2"], prior)
        report(title + " validation: calibrated / fused", calibrated, local_labels, names, prior)
        report(title + " evaluation: raw", evaluation, evaluation_labels, ["System 1", "System 2"], prior)
        report(title + " evaluation: calibrated / fused", eval_calibrated, evaluation_labels, names, prior)
        figures.append((title.lower().replace("-", "_") + "_calibration_fusion",
                        plot_example(title, raw, calibrated, local_labels,
                                     evaluation, eval_calibrated, evaluation_labels)))
    return figures


# 3. Restore the three fixed candidates selected in Lab 9. No base-model refit.

def read_cache(lab):
    path = LABS_FOLDER / f"lab{lab:02}" / "results/project_models.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_project_models():
    logistic, svm, gmm = [read_cache(lab) for lab in [7, 8, 9]]
    for cache in [logistic, svm]:
        for key in ["seed", "training_indices", "validation_indices", "training_data",
                    "training_labels", "validation_data", "validation_labels"]:
            if not np.array_equal(cache[key], gmm[key]):
                raise ValueError(f"Project caches do not agree on {key}; rerun Labs 7–9 with one split.")
    # These are the winning configurations recorded in our Lab 9 comparison.
    models = {
        "gmm_fake": unpack_gmm(gmm, "class0_1"),
        "gmm_genuine": unpack_gmm(gmm, "class1_16"),
        "logistic": {"w": logistic["quadratic_w"][5], "b": logistic["quadratic_b"][5],
                     "training_prior": logistic["quadratic_training_priors"][5]},
        "svm": {"alpha": svm["rbf2_alpha"][9], "coefficients": svm["rbf2_coefficients"][9],
                "training_data": svm["training_data"],
                "kernel": lambda first, second: rbf_kernel(first, second, gamma=float(svm["rbf_gammas"][2]), bias=1)}
    }
    scores = np.vstack([gmm["selected_gmm_scores"], gmm["selected_logistic_scores"], gmm["selected_svm_scores"]])
    if not np.allclose(score_project(models, gmm["validation_data"]), scores, rtol=0, atol=1e-9):
        raise ValueError("The restored models do not reproduce the Lab 9 selected scores.")
    print("Fixed base models: full GMM 1/16; quadratic LR lambda=0.0316228; RBF SVM gamma=exp(-2), C=31.6228")
    return models, scores, gmm["validation_labels"], logistic, gmm


def score_project(models, data):
    return np.vstack([
        logpdf_GMM(data, models["gmm_genuine"]) - logpdf_GMM(data, models["gmm_fake"]),
        binary_scores(models["logistic"], quadratic_features(data)),
        kernel_scores(models["svm"], data)
    ])


def calibrate_candidates(scores, labels, prior=0.1):
    inputs = [scores[index:index+1] for index in range(3)] + [scores]
    pooled = np.vstack([kfold_calibration(rows, labels, prior)[0] for rows in inputs])
    results = report("Five-fold project validation", pooled, labels, NAMES, prior)
    chosen = int(np.argmin([result["actual"] for result in results]))
    final_models = [fit_calibration(rows, labels, prior) for rows in inputs]
    print(f"\nDelivered system chosen by validation actDCF: {NAMES[chosen]}")
    return pooled, final_models, chosen


# 4. Optional audit: all 52 Lab 7 configurations, keeping the delivered choice fixed.

def prepare_logistic_audit(cache):
    candidates = []
    for group in ["linear", "reduced", "weighted", "quadratic"]:
        for index, regularization in enumerate(cache["lambdas"]):
            if group == "reduced":
                model = train_binary(cache["training_data"][:, ::50], cache["training_labels"][::50], regularization)
            else:
                model = {"w": cache[f"{group}_w"][index], "b": cache[f"{group}_b"][index],
                         "training_prior": cache[f"{group}_training_priors"][index]}
            data = quadratic_features(cache["validation_data"]) if group == "quadratic" else cache["validation_data"]
            minimum = cost(binary_scores(model, data), cache["validation_labels"], .1)["minimum"]
            candidates.append({"group": group, "lambda": regularization, "model": model, "validation_minimum": minimum})
    return candidates


def evaluate_logistic_audit(candidates, data, labels):
    quadratic = quadratic_features(data)
    for candidate in candidates:
        features = quadratic if candidate["group"] == "quadratic" else data
        candidate["evaluation_minimum"] = cost(binary_scores(candidate["model"], features), labels, .1)["minimum"]
    best = min(candidates, key=lambda row: row["evaluation_minimum"])
    selected = next(row for row in candidates if row["group"] == "quadratic" and np.isclose(row["lambda"], 10**-1.5))
    print("\nRetrospective LR audit (52 configurations; no change to the delivered system)")
    print(f"Earlier selected quadratic LR: eval minDCF {selected['evaluation_minimum']:.6f}")
    print(f"Lowest evaluation minDCF: {best['evaluation_minimum']:.6f}, {best['group']}, lambda={best['lambda']:g}")
    return candidates


def run_project(evaluation_file, save_folder=None):
    models, scores, labels, logistic_cache, gmm_cache = load_project_models()
    pooled, mappings, chosen = calibrate_candidates(scores, labels)
    # Prepare the optional audit models using training data BEFORE opening evaluation data.
    audit = prepare_logistic_audit(logistic_cache)
    print("Final calibration maps fitted on all 2,000 validation samples. Opening the final evaluation set.")
    evaluation, evaluation_labels = load_fingerprints(evaluation_file)
    raw_evaluation = score_project(models, evaluation)
    inputs = [raw_evaluation[index:index+1] for index in range(3)] + [raw_evaluation]
    calibrated = np.vstack([apply_calibration(model, rows) for model, rows in zip(mappings, inputs)])
    report("Final evaluation: original scores", raw_evaluation, evaluation_labels, NAMES[:3], .1)
    results = report("Final evaluation: calibrated / fused", calibrated, evaluation_labels, NAMES, .1)
    print(f"Delivered {NAMES[chosen]}: minDCF {results[chosen]['minimum']:.6f}; actDCF {results[chosen]['actual']:.6f}")
    print("Delivered confusion matrix (rows predicted, columns actual):\n", results[chosen]["matrix"])
    for prior in [.5, .9]:
        report("Final evaluation: alternative application", calibrated, evaluation_labels, NAMES, prior)
    audit = evaluate_logistic_audit(audit, evaluation, evaluation_labels)
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True)
        cache = {"application_prior": np.array(.1), "folds": np.array(5),
                 "fold_assignment": np.arange(labels.size) % 5, "names": np.array(NAMES),
                 "chosen_index": np.array(chosen), "chosen_name": np.array(NAMES[chosen]),
                 "development_raw_scores": scores, "development_calibrated_scores": pooled,
                 "development_labels": labels, "evaluation_raw_scores": raw_evaluation,
                 "evaluation_calibrated_scores": calibrated, "evaluation_labels": evaluation_labels,
                 "seed": gmm_cache["seed"], "validation_indices": gmm_cache["validation_indices"],
                 "audit_groups": np.array([row["group"] for row in audit]),
                 "audit_lambdas": np.array([row["lambda"] for row in audit]),
                 "audit_validation_minimum": np.array([row["validation_minimum"] for row in audit]),
                 "audit_evaluation_minimum": np.array([row["evaluation_minimum"] for row in audit])}
        for index, model in enumerate(mappings):
            cache[f"mapping{index}_w"] = model["w"]
            cache[f"mapping{index}_offset"] = np.array(model["offset"])
        for index, row in enumerate(audit):
            cache[f"audit{index}_w"] = row["model"]["w"]
            cache[f"audit{index}_b"] = np.array(row["model"]["b"])
            cache[f"audit{index}_training_prior"] = np.array(row["model"]["training_prior"])
        np.savez_compressed(save_folder / "project_models.npz", **cache)
        print("Saved final mappings, frozen choice, scores, and audit in one project_models.npz.")
    return [
        ("fingerprints_calibration_fusion", plot_project(scores, pooled, labels, raw_evaluation, calibrated, evaluation_labels)),
        ("fingerprints_delivered_system", plot_delivered(pooled, labels, calibrated, evaluation_labels, chosen)),
        ("fingerprints_logistic_audit", plot_audit(audit))
    ]


# 5. A few overview figures keep the output folder small.

def draw_curves(axis, scores, labels, name, color, log_odds, actual=True, minimum=True):
    act, low = bayes_error_curve(scores, labels, log_odds)
    if actual:
        axis.plot(log_odds, act, color=color, label=name + " actual")
    if minimum:
        axis.plot(log_odds, low, "--", color=color, label=name + " minimum")


def finish_axis(axis, title):
    axis.set(xlabel="Effective prior log odds", ylabel="Normalized DCF", title=title, ylim=(0, None))
    axis.axhline(1, color="gray", linestyle=":", linewidth=1)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)


def plot_example(title, raw, calibrated, labels, eval_raw, eval_cal, eval_labels):
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    log_odds = np.linspace(-3, 3, 61)
    for row, before, after, local_labels, partition in [
            (0, raw, calibrated, labels, "Validation"), (1, eval_raw, eval_cal, eval_labels, "Evaluation")]:
        for index in range(2):
            axis = axes[row, index]
            draw_curves(axis, before[index], local_labels, "Raw", "gray", log_odds)
            draw_curves(axis, after[index], local_labels, "Calibrated", COLORS[index], log_odds)
            finish_axis(axis, f"{partition}: system {index+1}")
        for index, name in enumerate(["System 1", "System 2", "Fusion"]):
            draw_curves(axes[row, 2], after[index], local_labels, name, COLORS[index], log_odds)
        finish_axis(axes[row, 2], partition + ": fusion")
    figure.suptitle(title + " calibration and fusion — training prior 0.2")
    return figure


def plot_project(raw, pooled, labels, eval_raw, calibrated, eval_labels):
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    log_odds = np.linspace(-4, 4, 81)
    for row, before, after, local_labels, partition in [
            (0, raw, pooled, labels, "Five-fold validation"),
            (1, eval_raw, calibrated, eval_labels, "Final evaluation")]:
        for index in range(3):
            draw_curves(axes[row, index], before[index], local_labels, "Original", "gray", log_odds)
            draw_curves(axes[row, index], after[index], local_labels, "Calibrated", COLORS[index], log_odds)
            finish_axis(axes[row, index], f"{partition}: {NAMES[index]}")
    figure.suptitle("Fingerprint calibration — training prior 0.1")
    return figure


def plot_delivered(development, development_labels, evaluation, evaluation_labels, chosen):
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    log_odds = np.linspace(-4, 4, 81)
    for row, scores, labels, partition in [(0, development, development_labels, "Five-fold validation"),
                                          (1, evaluation, evaluation_labels, "Final evaluation")]:
        draw_curves(axes[row, 0], scores[chosen], labels, NAMES[chosen], COLORS[chosen], log_odds)
        finish_axis(axes[row, 0], partition + ": delivered " + NAMES[chosen])
        for index, name in enumerate(NAMES):
            draw_curves(axes[row, 1], scores[index], labels, name, COLORS[index], log_odds, minimum=False)
            draw_curves(axes[row, 2], scores[index], labels, name, COLORS[index], log_odds, actual=False)
        finish_axis(axes[row, 1], partition + ": actual DCF")
        finish_axis(axes[row, 2], partition + ": minimum DCF")
    figure.suptitle("Fingerprint calibration and fusion — delivered choice fixed using validation")
    return figure


def plot_audit(audit):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for group, color in zip(["linear", "reduced", "weighted", "quadratic"], COLORS):
        rows = [row for row in audit if row["group"] == group]
        lambdas = [row["lambda"] for row in rows]
        axes[0].semilogx(lambdas, [row["validation_minimum"] for row in rows], "o-", color=color, label=group)
        axes[1].semilogx(lambdas, [row["evaluation_minimum"] for row in rows], "o-", color=color, label=group)
    for axis, title in zip(axes, ["Earlier validation", "Final evaluation (retrospective)"]):
        axis.set(xlabel="Regularization lambda", ylabel="Minimum normalized DCF", title=title, ylim=(0, None))
        axis.axvline(10**-1.5, linestyle=":", color="gray", label="Selected quadratic lambda")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Logistic-regression audit: all 52 earlier configurations — prior 0.1")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="store_true", help="calibrate saved fingerprint models and evaluate them")
    parser.add_argument("--evaluation", type=Path, help="custom project evaluation data; never used for training")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    if args.evaluation is not None and not args.project:
        parser.error("--evaluation requires --project")
    try:
        if args.project:
            evaluation_file = args.evaluation or LABS_FOLDER / "lab02/data/evaldata.txt"
            figures = run_project(evaluation_file, folder / "results" if args.save else None)
        else:
            figures = run_examples(folder / "data")
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    if args.save:
        (folder / "results").mkdir(exist_ok=True)
        for name, figure in figures:
            path = folder / "results" / f"{name}.png"
            figure.savefig(path, dpi=140)
            print("Saved:", path)
    if not args.no_show:
        plt.show()
    for name, figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
