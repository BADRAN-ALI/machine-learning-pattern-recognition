"""Lab 6: Bayes decisions, detection costs, ROC curves, and model evaluation."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

LABS_FOLDER = Path(__file__).resolve().parents[1]
for folder in ["lab02", "lab03", "lab05"]:
    sys.path.insert(0, str(LABS_FOLDER / folder))
from lab02 import load_iris, load_fingerprints, COLORS
from lab03 import split_data
from lab05 import MODELS, fit_classifier, class_log_scores, infer_log, compute_llr, class_priors

APPLICATIONS = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]


# 1. Confusion matrices and multiclass Bayes decisions.

def confusion_matrix(predictions, labels, class_count):
    """Rows are predicted classes; columns are actual classes, as in this sheet."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("Predictions and labels must be matching, nonempty vectors.")
    for values in [predictions, labels]:
        if (not np.issubdtype(values.dtype, np.integer) or
                np.any(values < 0) or np.any(values >= class_count)):
            raise ValueError("Class labels must be integers from 0 to class_count - 1.")
    matrix = np.zeros((class_count, class_count), dtype=int)
    for predicted, actual in zip(predictions, labels):
        matrix[predicted, actual] += 1
    return matrix


def check_costs(costs, class_count):
    costs = np.asarray(costs, dtype=float)
    if (costs.shape != (class_count, class_count) or
            not np.isfinite(costs).all() or np.any(costs < 0)):
        raise ValueError("Costs must be a finite, nonnegative square matrix.")
    return costs


def bayes_multiclass(log_likelihoods, priors, costs):
    """Choose the decision with the smallest posterior expected cost."""
    log_likelihoods = np.asarray(log_likelihoods, dtype=float)
    if (log_likelihoods.ndim != 2 or log_likelihoods.shape[1] == 0 or
            not np.isfinite(log_likelihoods).all()):
        raise ValueError("Log likelihoods must be a finite class-by-sample matrix.")
    costs = check_costs(costs, log_likelihoods.shape[0])
    posterior = infer_log(log_likelihoods, priors)["posterior"]
    expected_cost = costs @ posterior
    return np.argmin(expected_cost, axis=0).astype(np.int32)


def multiclass_dcf(matrix, priors, costs, normalize=True):
    priors = class_priors(matrix.shape[0], priors)
    costs = check_costs(costs, matrix.shape[0])
    counts = matrix.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Each actual class needs at least one sample.")
    # Normalize each column: P(predicted class i | actual class j).
    rates = matrix / counts[None, :]
    risk = float(priors @ (costs * rates).sum(axis=0))
    if not normalize:
        return risk
    dummy_cost = float(np.min(costs @ priors))
    if dummy_cost <= 0:
        raise ValueError("Normalized cost needs a positive dummy cost.")
    return risk / dummy_cost


# 2. Binary decisions and actual detection cost.

def check_application(prior, cost_fn, cost_fp):
    values = np.array([prior, cost_fn, cost_fp], dtype=float)
    if not np.isfinite(values).all() or not 0 < prior < 1 or min(cost_fn, cost_fp) <= 0:
        raise ValueError("Use a prior strictly between 0 and 1 and positive finite costs.")


def effective_prior(prior, cost_fn, cost_fp):
    check_application(prior, cost_fn, cost_fp)
    return prior * cost_fn / (prior * cost_fn + (1 - prior) * cost_fp)


def bayes_threshold(prior, cost_fn=1, cost_fp=1):
    check_application(prior, cost_fn, cost_fp)
    return float(np.log1p(-prior) + np.log(cost_fp) - np.log(prior) - np.log(cost_fn))


def binary_decisions(scores, threshold):
    # The sheet uses strict >: a score equal to the threshold belongs to class 0.
    return (np.asarray(scores) > threshold).astype(np.int32)


def binary_rates(matrix):
    if matrix.shape != (2, 2) or np.any(matrix.sum(axis=0) == 0):
        raise ValueError("A binary confusion matrix must contain both actual classes.")
    fnr = matrix[0, 1] / matrix[:, 1].sum()
    fpr = matrix[1, 0] / matrix[:, 0].sum()
    return float(fnr), float(fpr)


def cost_from_rates(fnr, fpr, prior, cost_fn=1, cost_fp=1, normalize=True):
    check_application(prior, cost_fn, cost_fp)
    cost = prior * cost_fn * fnr + (1 - prior) * cost_fp * fpr
    if normalize:
        cost = cost / min(prior * cost_fn, (1 - prior) * cost_fp)
    return cost


def binary_dcf(matrix, prior, cost_fn=1, cost_fp=1, normalize=True):
    fnr, fpr = binary_rates(matrix)
    return float(cost_from_rates(fnr, fpr, prior, cost_fn, cost_fp, normalize))


# 3. Sweep thresholds once; reuse the rates for minDCF, ROC, and Bayes plots.

def threshold_sweep(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if (scores.ndim != 1 or scores.shape != labels.shape or scores.size == 0 or
            not np.isfinite(scores).all() or not np.array_equal(np.unique(labels), [0, 1])):
        raise ValueError("Use finite score and binary-label vectors containing both classes.")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    # Increasing the threshold moves samples from predicted 1 to predicted 0.
    # Equal scores must move together; only the last index of each tie is used.
    last_indices = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    false_negatives = np.cumsum(sorted_labels == 1)[last_indices]
    true_negatives = np.cumsum(sorted_labels == 0)[last_indices]
    thresholds = np.r_[-np.inf, sorted_scores[last_indices], np.inf]
    fnr = np.r_[0.0, false_negatives / positives, 1.0]
    fpr = np.r_[1.0, (negatives - true_negatives) / negatives, 0.0]
    return thresholds, fnr, fpr


def minimum_dcf(sweep, prior, cost_fn=1, cost_fp=1):
    thresholds, fnr, fpr = sweep
    costs = cost_from_rates(fnr, fpr, prior, cost_fn, cost_fp)
    best = int(np.argmin(costs))
    return float(costs[best]), float(thresholds[best])


def evaluate_scores(scores, labels, applications):
    sweep = threshold_sweep(scores, labels)
    results = []
    for prior, cost_fn, cost_fp in applications:
        threshold = bayes_threshold(prior, cost_fn, cost_fp)
        matrix = confusion_matrix(binary_decisions(scores, threshold), labels, 2)
        minimum, best_threshold = minimum_dcf(sweep, prior, cost_fn, cost_fp)
        actual = binary_dcf(matrix, prior, cost_fn, cost_fp)
        results.append({"application": (prior, cost_fn, cost_fp), "matrix": matrix,
                        "threshold": threshold, "dcf": binary_dcf(matrix, prior, cost_fn, cost_fp, False),
                        "actual": actual, "minimum": minimum, "best_threshold": best_threshold,
                        "gap": actual - minimum, "relative_gap": 100 * (actual - minimum) / minimum
                        if minimum > 0 else np.nan})
    return results


def bayes_error_curve(scores, labels, log_odds):
    sweep = threshold_sweep(scores, labels)
    actual, minimum = [], []
    for prior in expit(log_odds):
        matrix = confusion_matrix(binary_decisions(scores, bayes_threshold(prior)), labels, 2)
        actual.append(binary_dcf(matrix, prior))
        minimum.append(minimum_dcf(sweep, prior)[0])
    return np.array(actual), np.array(minimum)


def print_binary_results(name, results):
    print(f"\n{name}: rows predicted, columns actual")
    print("(prior, Cfn, Cfp)    threshold    DCF      actDCF   minDCF   gap (%)")
    for result in results:
        print(f"{str(result['application']):<20} {result['threshold']:8.3f} "
              f"{result['dcf']:8.3f} {result['actual']:8.3f} {result['minimum']:8.3f} "
              f"{result['relative_gap']:9.2f}")
        print(result["matrix"])


# 4. Three overview figures instead of a separate file for each experiment.

def draw_confusion(axis, matrix, title):
    axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    axis.set_xticks(range(matrix.shape[0]))
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_xlabel("Actual class")
    axis.set_ylabel("Predicted class")
    axis.set_title(title)


def plot_multiclass(iris_matrices, supplied_results):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row, (model, matrix) in enumerate(iris_matrices):
        draw_confusion(axes[row, 0], matrix, f"Iris: {model}")
    for row, results in enumerate(supplied_results):
        for column, result in enumerate(results, start=1):
            draw_confusion(axes[row, column], result["matrix"],
                           f"Recognizer {row + 1}: {result['name']}\n"
                           f"Normalized cost = {result['normalized']:.3f}")
    figure.suptitle("Multiclass Bayes decisions (class IDs 0, 1, 2)")
    return figure


def plot_binary(scores_list, labels, results_list):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    log_odds = np.linspace(-3, 3, 21)
    for index, scores in enumerate(scores_list):
        name = f"Recognizer {index + 1}"
        _, fnr, fpr = threshold_sweep(scores, labels)
        axes[0, 0].plot(fpr[::-1], (1 - fnr)[::-1], color=COLORS[index], label=name)
        actual, minimum = bayes_error_curve(scores, labels, log_odds)
        axes[0, 1].plot(log_odds, actual, color=COLORS[index], label=f"{name}: actual")
        axes[0, 1].plot(log_odds, minimum, "--", color=COLORS[index], label=f"{name}: minimum")
        axis = axes[1, index]
        positions = np.arange(len(APPLICATIONS))
        axis.bar(positions - 0.18, [r["actual"] for r in results_list[index]], 0.36,
                 color=COLORS[index], label="Actual DCF")
        axis.bar(positions + 0.18, [r["minimum"] for r in results_list[index]], 0.36,
                 color=COLORS[index], alpha=0.45, label="Minimum DCF")
        axis.set_xticks(positions, [str(application) for application in APPLICATIONS], rotation=10)
        axis.set_xlabel("(positive prior, false-negative cost, false-positive cost)")
        axis.set_ylabel("Normalized DCF")
        axis.set_title(name)
        axis.axhline(1, color="gray", linestyle=":", label="Best constant decision")
        axis.legend()
    axes[0, 0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves",
                   xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].set(xlabel="Effective prior log odds", ylabel="Normalized DCF", title="Bayes error curves")
    axes[0, 1].axhline(1, color="gray", linestyle=":")
    for axis in axes[0]:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
    figure.suptitle("Binary score evaluation: 802 supplied labeled samples")
    return figure


def plot_project(scores_list, labels):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    log_odds = np.linspace(-4, 4, 41)
    for model, scores, color in zip(MODELS, scores_list, COLORS):
        actual, minimum = bayes_error_curve(scores, labels, log_odds)
        axes[0].plot(log_odds, minimum, color=color, label=model)
        axes[1].plot(log_odds, actual, color=color, label=model)
    for axis, title in zip(axes, ["Discrimination: minimum DCF", "Bayes threshold: actual DCF"]):
        axis.set(xlabel="Effective prior log odds", title=title, ylim=(0, None))
        axis.axhline(1, color="gray", linestyle=":")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Normalized DCF")
    figure.suptitle("Fingerprint validation: all 6 features, no preprocessing")
    return figure


# 5. Iris, supplied recognizer scores, and the fingerprint project.

def run_lab(iris_data, iris_labels, seed, data_folder, check_iris=True):
    training, validation = split_data(iris_data, iris_labels, seed)
    iris_matrices = []
    expected_iris = [np.array([[19, 0, 0], [0, 15, 0], [0, 2, 14]]),
                     np.array([[19, 0, 0], [0, 16, 0], [0, 1, 14]])]
    uniform_priors = np.full(3, 1 / 3)
    uniform_costs = np.ones((3, 3)) - np.eye(3)
    for index, model in enumerate(MODELS[:2]):
        parameters = fit_classifier(*training, model)
        scores, classes = class_log_scores(validation[0], parameters)
        decisions = bayes_multiclass(scores, uniform_priors, uniform_costs)
        matrix = confusion_matrix(classes[decisions], validation[1], 3)
        iris_matrices.append((model, matrix))
        print(f"\nIris {model} (rows predicted, columns actual):\n{matrix}")
        if check_iris and seed == 0 and not np.array_equal(matrix, expected_iris[index]):
            raise ValueError(f"Iris {model} differs from the sheet's confusion matrix.")

    binary_labels = np.load(data_folder / "evalset_labels_binary.npy", allow_pickle=False)
    multiclass_labels = np.load(data_folder / "evalset_labels.npy", allow_pickle=False)
    scores_list, binary_results, multiclass_results = [], [], []
    priors = np.array([0.3, 0.4, 0.3])
    costs = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    expected_binary = [np.array([[293, 96], [109, 304]]), np.array([[271, 80], [131, 320]]),
                       np.array([[257, 75], [145, 325]]), np.array([[302, 113], [100, 287]])]
    expected_multi = [[np.array([[210, 113, 61], [137, 191, 111], [53, 98, 230]]),
                       np.array([[205, 111, 56], [145, 199, 121], [50, 92, 225]])],
                      [None, np.array([[216, 77, 31], [146, 236, 143], [38, 89, 228]])]]
    for index, suffix in enumerate(["", "_2"]):
        scores = np.load(data_folder / f"evalset_llr_binary{suffix}.npy", allow_pickle=False)
        scores_list.append(scores)
        results = evaluate_scores(scores, binary_labels, APPLICATIONS)
        binary_results.append(results)
        print_binary_results(f"Binary recognizer {index + 1}", results)
        if index == 0:
            for result, expected in zip(results, expected_binary):
                if not np.array_equal(result["matrix"], expected):
                    raise ValueError("Binary confusion matrix differs from the sheet.")

        log_likelihoods = np.load(data_folder / f"evalset_ll{suffix}.npy", allow_pickle=False)
        results = []
        for app_index, (name, app_priors, app_costs) in enumerate([
                ("Uniform costs/priors", uniform_priors, uniform_costs),
                ("Nonuniform costs/priors", priors, costs)]):
            predictions = bayes_multiclass(log_likelihoods, app_priors, app_costs)
            matrix = confusion_matrix(predictions, multiclass_labels, 3)
            result = {"name": name, "matrix": matrix,
                      "dcf": multiclass_dcf(matrix, app_priors, app_costs, False),
                      "normalized": multiclass_dcf(matrix, app_priors, app_costs)}
            results.append(result)
            print(f"\nMulticlass recognizer {index + 1}: {name}\n{matrix}")
            print(f"Cost: {result['dcf']:.3f}; normalized: {result['normalized']:.3f}")
            expected = expected_multi[index][app_index]
            if expected is not None and not np.array_equal(matrix, expected):
                raise ValueError("Multiclass confusion matrix differs from the sheet.")
        multiclass_results.append(results)
    print("\nAll applicable confusion-matrix references match.")
    print("minDCF selects a threshold using these labels; it is an oracle comparison, not calibration.")
    return [("binary_evaluation", plot_binary(scores_list, binary_labels, binary_results)),
            ("multiclass_decisions", plot_multiclass(iris_matrices, multiclass_results))]


def run_project(data, labels, seed):
    training, validation = split_data(data, labels, seed)
    print(f"Project split: {training[1].size} training, {validation[1].size} validation")
    print("\nApplication -> equivalent prior with unit costs:")
    for application in [(0.5, 1, 1), (0.9, 1, 1), (0.1, 1, 1), (0.5, 1, 9), (0.5, 9, 1)]:
        print(application, "->", f"{effective_prior(*application):.1f}")
    applications = [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]
    scores_list = []
    for model in MODELS:
        parameters = fit_classifier(*training, model)
        scores, classes = compute_llr(validation[0], parameters)
        scores_list.append(scores)
        results = evaluate_scores(scores, validation[1], applications)
        print_binary_results(model, results)
    print("\nThe minimum-cost threshold is selected on validation labels for comparison only.")
    print("Relative gap = 100 * (actual DCF - minimum DCF) / minimum DCF.")
    return [("fingerprints_bayes_evaluation", plot_project(scores_list, validation[1]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="optional custom Iris CSV or project data file")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    lab_folder = Path(__file__).resolve().parent
    filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data" / (
        "traindata.txt" if args.project else "iris.csv")
    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            figures = run_project(data, labels, args.seed)
        else:
            data, labels = load_iris(filename)
            figures = run_lab(data, labels, args.seed, lab_folder / "data", args.filename is None)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    if args.save:
        folder = lab_folder / "results"
        folder.mkdir(exist_ok=True)
        for name, figure in figures:
            path = folder / f"{name}.png"
            figure.savefig(path, dpi=140)
            print("Saved:", path)
    if not args.no_show:
        plt.show()
    for name, figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
