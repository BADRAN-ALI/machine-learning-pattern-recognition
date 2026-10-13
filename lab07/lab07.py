"""Lab 7: logistic regression, regularization, and fingerprint classification."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import expit, logsumexp

LABS_FOLDER = Path(__file__).resolve().parents[1]
for folder in ["lab02", "lab03", "lab05", "lab06"]:
    sys.path.insert(0, str(LABS_FOLDER / folder))
from lab02 import load_iris, load_fingerprints, COLORS
from lab03 import split_data, fit_pca
from lab05 import MODELS, fit_classifier, compute_llr
from lab06 import evaluate_scores, confusion_matrix, draw_confusion

IRIS_LAMBDAS = [1e-3, 1e-1, 1.0]
PROJECT_LAMBDAS = np.logspace(-4, 2, 13)


# 1. Numerical optimization with approximate and explicit gradients.

def toy_objective(vector):
    y, z = vector
    return (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2


def toy_objective_gradient(vector):
    y, z = vector
    gradient = np.array([2 * (y + 3) + np.cos(y), 2 * (z + 1)])
    return toy_objective(vector), gradient


def run_optimization():
    results = []
    for name, objective, numerical in [("Numerical gradient", toy_objective, True),
                                       ("Explicit gradient", toy_objective_gradient, False)]:
        point, value, info = fmin_l_bfgs_b(objective, np.zeros(2), approx_grad=numerical)
        if info["warnflag"] != 0:
            raise ValueError(f"Toy optimization did not converge: {info['task']}")
        print(f"{name}: x = {point}, f = {value:.9f}, calls = {info['funcalls']}")
        results.append((point, value, info))
    return results


# 2. Binary logistic regression: stable loss, explicit gradient, unpenalized bias.

def check_training(data, labels, regularization):
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels)
    if (data.ndim != 2 or min(data.shape) == 0 or labels.shape != (data.shape[1],) or
            not np.isfinite(data).all() or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Use a finite feature-by-sample matrix and one integer label per column.")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Regularization must be finite and nonnegative.")
    return data, labels


def make_binary_objective(data, labels, regularization, target_prior=None):
    data, labels = check_training(data, labels, regularization)
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Binary training requires both classes, labeled 0 and 1.")
    signs = 2 * labels - 1
    if target_prior is None:
        weights = np.full(labels.size, 1 / labels.size)
    else:
        if not np.isfinite(target_prior) or not 0 < target_prior < 1:
            raise ValueError("The target training prior must be strictly between zero and one.")
        weights = np.where(labels == 1, target_prior / np.sum(labels == 1),
                           (1 - target_prior) / np.sum(labels == 0))

    def objective(vector):
        w, b = vector[:-1], vector[-1]
        scores = w @ data + b
        losses = np.logaddexp(0, -signs * scores)
        value = 0.5 * regularization * (w @ w) + weights @ losses
        # expit avoids overflow for very large positive or negative margins.
        score_gradient = -signs * expit(-signs * scores) * weights
        gradient_w = regularization * w + data @ score_gradient
        gradient_b = score_gradient.sum()
        return float(value), np.r_[gradient_w, gradient_b]

    return objective


def train_binary(data, labels, regularization, target_prior=None, numerical=False):
    objective = make_binary_objective(data, labels, regularization, target_prior)
    function = (lambda vector: objective(vector)[0]) if numerical else objective
    vector, value, info = fmin_l_bfgs_b(function, np.zeros(data.shape[0] + 1),
                                     approx_grad=numerical, maxfun=15000, maxiter=15000)
    if info["warnflag"] != 0:
        raise ValueError(f"Binary optimizer did not converge: {info['task']}")
    training_prior = float(np.mean(labels == 1)) if target_prior is None else float(target_prior)
    return {"w": vector[:-1], "b": float(vector[-1]), "loss": float(value),
            "training_prior": training_prior, "regularization": regularization,
            "funcalls": info["funcalls"], "iterations": info["nit"]}


def binary_scores(model, data, as_llr=True):
    scores = model["w"] @ data + model["b"]
    if as_llr:
        prior = model["training_prior"]
        scores = scores - np.log(prior / (1 - prior))
    return scores


def binary_metrics(model, validation, application_prior):
    data, labels = validation
    scores = binary_scores(model, data)
    result = evaluate_scores(scores, labels, [(application_prior, 1, 1)])[0]
    # The sheet's ordinary classification error uses the raw score > 0.
    predictions = (binary_scores(model, data, as_llr=False) > 0).astype(np.int32)
    result.update({"model": model, "scores": scores, "loss": model["loss"],
                   "error_rate": float(np.mean(predictions != labels))})
    return result


def quadratic_features(data):
    """For each column x, stack vec(x x.T) and x: D features become D*D + D."""
    dimensions, samples = data.shape
    products = (data[:, None, :] * data[None, :, :]).reshape(dimensions * dimensions, samples)
    return np.vstack([products, data])


# 3. Optional multiclass model: softmax cross-entropy and its gradient.

def make_multiclass_objective(data, labels, regularization):
    data, labels = check_training(data, labels, regularization)
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("Multiclass training needs at least two classes.")
    dimensions, samples = data.shape
    targets = np.zeros((classes.size, samples))
    targets[np.searchsorted(classes, labels), np.arange(samples)] = 1

    def objective(vector):
        w = vector[:dimensions * classes.size].reshape(dimensions, classes.size)
        b = vector[dimensions * classes.size:, None]
        scores = w.T @ data + b
        log_posterior = scores - logsumexp(scores, axis=0, keepdims=True)
        value = 0.5 * regularization * np.sum(w * w) - np.sum(targets * log_posterior) / samples
        score_gradient = (np.exp(log_posterior) - targets) / samples
        gradient_w = regularization * w + data @ score_gradient.T
        gradient_b = score_gradient.sum(axis=1)
        return float(value), np.r_[gradient_w.ravel(), gradient_b]

    return objective, classes


def train_multiclass(data, labels, regularization):
    objective, classes = make_multiclass_objective(data, labels, regularization)
    size = (data.shape[0] + 1) * classes.size
    vector, value, info = fmin_l_bfgs_b(objective, np.zeros(size), maxfun=15000, maxiter=15000)
    if info["warnflag"] != 0:
        raise ValueError(f"Multiclass optimizer did not converge: {info['task']}")
    split = data.shape[0] * classes.size
    return {"w": vector[:split].reshape(data.shape[0], classes.size),
            "b": vector[split:, None], "classes": classes, "loss": float(value)}


def multiclass_log_posterior(model, data):
    scores = model["w"].T @ data + model["b"]
    return scores - logsumexp(scores, axis=0, keepdims=True)


# 4. Overview figures and result tables.

def print_binary_results(name, lambdas, results):
    print(f"\n{name}")
    print("lambda       training loss    error (%)    actual DCF    minimum DCF")
    for regularization, result in zip(lambdas, results):
        print(f"{regularization:10.4g} {result['loss']:15.8f} "
              f"{100 * result['error_rate']:12.2f} {result['actual']:13.4f} {result['minimum']:14.4f}")


def plot_iris(binary_results, weighted_results, multiclass_results, labels):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for index, (name, results) in enumerate([("Standard binary: prior 0.5", binary_results),
                                            ("Weighted binary: prior 0.8", weighted_results)]):
        axis = axes[0, index]
        axis.semilogx(IRIS_LAMBDAS, [r["actual"] for r in results], "o-", label="Actual DCF")
        axis.semilogx(IRIS_LAMBDAS, [r["minimum"] for r in results], "o--", label="Minimum DCF")
        axis.set(xlabel="Regularization lambda", ylabel="Normalized DCF", title=name)
        axis.legend()
        axis.grid(alpha=0.2)
    axis = axes[0, 2]
    axis.semilogx(IRIS_LAMBDAS, [100 * r["error_rate"] for r in multiclass_results], "o-", color=COLORS[2])
    axis.set(xlabel="Regularization lambda", ylabel="Error rate (%)", title="Multiclass validation error")
    axis.grid(alpha=0.2)
    for axis, regularization, result in zip(axes[1], IRIS_LAMBDAS, multiclass_results):
        matrix = confusion_matrix(result["predictions"], labels, 3)
        draw_confusion(axis, matrix, f"Multiclass: lambda {regularization:g}\n"
                       f"{100 * result['error_rate']:.1f}% error")
    figure.suptitle("Iris logistic regression: seed-0 validation split")
    return figure


def plot_project(experiments):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, layout="constrained")
    for axis, (name, results) in zip(axes.ravel(), experiments.items()):
        axis.semilogx(PROJECT_LAMBDAS, [r["actual"] for r in results], "o-", label="Actual DCF")
        axis.semilogx(PROJECT_LAMBDAS, [r["minimum"] for r in results], "o--", label="Minimum DCF")
        axis.axhline(1, color="gray", linestyle=":", label="Best constant decision")
        axis.set(xlabel="Regularization lambda", ylabel="Normalized DCF", title=name)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    axes[0, 0].set_ylim(bottom=0)
    figure.suptitle("Fingerprint logistic regression: application prior 0.1")
    return figure


def plot_model_comparison(comparisons):
    figure, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    positions = np.arange(len(comparisons))
    axis.bar(positions - 0.18, [item["minimum"] for item in comparisons], 0.36, label="Minimum DCF")
    axis.bar(positions + 0.18, [item["actual"] for item in comparisons], 0.36, label="Actual DCF")
    axis.set_xticks(positions, [item["name"] for item in comparisons], rotation=15, ha="right")
    axis.set(ylabel="Normalized DCF", title="Full-training models: application prior 0.1")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return figure


# 5. Exercises, reference checks, and the fingerprint project.

def run_iris(data, labels, seed, check_references=True):
    run_optimization()
    keep = labels != 0
    binary_data = data[:, keep]
    binary_labels = np.where(labels[keep] == 1, 1, 0).astype(np.int32)
    training, validation = split_data(binary_data, binary_labels, seed)
    print(f"\nBinary Iris: {training[1].size} training, {validation[1].size} validation")
    print("Labels: Virginica = 0; Versicolor = 1 (this sheet reverses the earlier mapping).")
    standard, weighted = [], []
    for regularization in IRIS_LAMBDAS:
        model = train_binary(*training, regularization)
        standard.append(binary_metrics(model, validation, 0.5))
        numerical = train_binary(*training, regularization, numerical=True)
        difference = np.max(np.abs(binary_scores(model, validation[0]) -
                                   binary_scores(numerical, validation[0])))
        print(f"lambda {regularization:g}: explicit/numerical calls = "
              f"{model['funcalls']}/{numerical['funcalls']}; maximum score difference = {difference:.3g}")
        model = train_binary(*training, regularization, target_prior=0.8)
        weighted.append(binary_metrics(model, validation, 0.8))
    print_binary_results("Standard binary Iris (application prior 0.5)", IRIS_LAMBDAS, standard)
    print_binary_results("Prior-weighted binary Iris (training/application prior 0.8)", IRIS_LAMBDAS, weighted)

    empirical_prior = np.mean(training[1] == 1)
    original = standard[0]["model"]
    equivalent = train_binary(*training, IRIS_LAMBDAS[0], target_prior=empirical_prior)
    delta = np.max(np.abs(binary_scores(original, validation[0]) - binary_scores(equivalent, validation[0])))
    print(f"\nWeighting at empirical prior {empirical_prior:.6f}: maximum score difference = {delta:.3g}")

    training, validation = split_data(data, labels, seed)
    multiclass = []
    print("\nOptional multiclass Iris\nlambda       training loss    error (%)")
    for regularization in IRIS_LAMBDAS:
        model = train_multiclass(*training, regularization)
        log_posterior = multiclass_log_posterior(model, validation[0])
        predictions = model["classes"][np.argmax(log_posterior, axis=0)]
        result = {"model": model, "loss": model["loss"], "predictions": predictions,
                  "error_rate": float(np.mean(predictions != validation[1]))}
        multiclass.append(result)
        print(f"{regularization:10.4g} {result['loss']:15.8f} {100 * result['error_rate']:12.2f}")
    if check_references and seed == 0:
        expected_standard = [[.1100009, .0625, .1181], [.4539407, .0556, .1111], [.6316436, .1111, .1667]]
        expected_weighted = [[.09401035, .1667, .2222], [.3606261, .0556, .7222], [.4724715, .1111, 1.0]]
        for results, expected in [(standard, expected_standard), (weighted, expected_weighted)]:
            calculated = [[r["loss"], r["minimum"], r["actual"]] for r in results]
            if not np.allclose(calculated, expected, rtol=0, atol=5e-5):
                raise ValueError("Binary logistic-regression results differ from the sheet.")
        if not np.allclose([r["loss"] for r in multiclass], [.0969097, .500591, .821155], rtol=0, atol=5e-6):
            raise ValueError("Multiclass training losses differ from the sheet.")
        if not np.array_equal([np.sum(r["predictions"] != validation[1]) for r in multiclass], [2, 3, 9]):
            raise ValueError("Multiclass error counts differ from the sheet.")
        print("\nAll Iris objective and DCF references match the sheet's rounding.")
    else:
        print("\nSheet references skipped: they use the default Iris data and seed 0.")
    return [("iris_logistic_regression", plot_iris(standard, weighted, multiclass, validation[1]))]


def run_project(data, labels, seed, save_folder=None):
    training, validation = split_data(data, labels, seed)
    reduced = (training[0][:, ::50], training[1][::50])
    print(f"Project: {training[1].size} full-training, {reduced[1].size} reduced-training, "
          f"{validation[1].size} validation samples")
    print("Reduced-training class counts:", np.bincount(reduced[1], minlength=2))
    experiments = {}
    cache = {"seed": np.array(seed), "lambdas": PROJECT_LAMBDAS,
             "training_data": training[0], "training_labels": training[1],
             "validation_data": validation[0],
             "validation_labels": validation[1], "application_prior": np.array(0.1),
             "feature_count": np.array(data.shape[0])}
    # Preserve split indices so later calibration labs can align these scores.
    indices = np.random.RandomState(seed).permutation(labels.size)
    split = training[1].size
    cache.update({"training_indices": indices[:split], "validation_indices": indices[split:]})
    configurations = [("Linear: full training", "linear", training, validation, None),
                      ("Linear: 1 in 50 training", "reduced", reduced, validation, None),
                      ("Weighted linear: prior 0.1", "weighted", training, validation, 0.1),
                      ("Quadratic: full training", "quadratic",
                       (quadratic_features(training[0]), training[1]),
                       (quadratic_features(validation[0]), validation[1]), None)]
    comparisons = []
    for name, key, train, val, target_prior in configurations:
        results = []
        for regularization in PROJECT_LAMBDAS:
            model = train_binary(*train, regularization, target_prior)
            results.append(binary_metrics(model, val, 0.1))
        experiments[name] = results
        print_binary_results(name, PROJECT_LAMBDAS, results)
        best = min(results, key=lambda result: result["minimum"])
        best_actual = min(results, key=lambda result: result["actual"])
        print(f"Best minDCF: {best['minimum']:.6f} at lambda {best['model']['regularization']:g}; "
              f"its actual DCF: {best['actual']:.6f}")
        print(f"Best actual DCF: {best_actual['actual']:.6f} at lambda "
              f"{best_actual['model']['regularization']:g}")
        if key != "reduced":
            cache[f"{key}_w"] = np.vstack([r["model"]["w"] for r in results])
            cache[f"{key}_b"] = np.array([r["model"]["b"] for r in results])
            cache[f"{key}_training_priors"] = np.array([r["model"]["training_prior"] for r in results])
            cache[f"{key}_validation_llr"] = np.vstack([r["scores"] for r in results])
            cache[f"{key}_training_loss"] = np.array([r["loss"] for r in results])
            comparisons.append({"name": name.split(":")[0], "minimum": best["minimum"],
                                "actual": best["actual"]})

    # Include the feature subsets and PCA variants explored in Lab 5.
    transformations = [("all6", "All 6", np.eye(data.shape[0]))]
    for key, name, selected in [("features1_4", "Features 1-4", [0, 1, 2, 3]),
                                 ("features1_2", "Features 1-2", [0, 1]),
                                 ("features3_4", "Features 3-4", [2, 3])]:
        transformations.append((key, name, np.eye(data.shape[0])[:, selected]))
    for dimensions in range(1, data.shape[0] + 1):
        projection, _ = fit_pca(training[0], dimensions)
        transformations.append((f"pca{dimensions}", f"PCA {dimensions}", projection))
    cache["gaussian_names"] = np.array(MODELS)
    cache["gaussian_configurations"] = np.array([key for key, _, _ in transformations])
    cache["gaussian_classes"] = np.array([0, 1], dtype=np.int32)
    for model_name in MODELS:
        candidates = []
        print(f"\n{model_name}: earlier feature and PCA configurations")
        for key, name, projection in transformations:
            parameters = fit_classifier(projection.T @ training[0], training[1], model_name)
            scores, classes = compute_llr(projection.T @ validation[0], parameters)
            result = evaluate_scores(scores, validation[1], [(0.1, 1, 1)])[0]
            result["configuration"] = name
            candidates.append(result)
            print(f"{name:<15} minimum {result['minimum']:.6f}; actual {result['actual']:.6f}")
            prefix = f"{model_name.lower()}_{key}"
            cache[f"{prefix}_projection"] = projection
            cache[f"{prefix}_means"] = np.stack([parameters[int(c)][0] for c in classes])
            cache[f"{prefix}_covariances"] = np.stack([parameters[int(c)][1] for c in classes])
            cache[f"{prefix}_validation_llr"] = scores
        best = min(candidates, key=lambda item: item["minimum"])
        comparisons.append({"name": f"{model_name} ({best['configuration']})",
                            "minimum": best["minimum"], "actual": best["actual"]})
    print("\nFull-training comparison (LR lambda selected by minimum DCF):")
    for result in sorted(comparisons, key=lambda item: item["minimum"]):
        print(f"{result['name']:<20} minimum {result['minimum']:.6f}; actual {result['actual']:.6f}")
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True)
        path = save_folder / "project_models.npz"
        np.savez_compressed(path, **cache)
        print("Saved models, priors, split indices, and validation LLRs:", path)
    print("\nLambda selection and minDCF use validation labels; these are validation comparisons.")
    return [("fingerprints_regularization", plot_project(experiments)),
            ("fingerprints_model_comparison", plot_model_comparison(comparisons))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="optional custom Iris CSV or project file")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    lab_folder = Path(__file__).resolve().parent
    results_folder = lab_folder / "results"
    filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data" / (
        "traindata.txt" if args.project else "iris.csv")
    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            figures = run_project(data, labels, args.seed, results_folder if args.save else None)
        else:
            data, labels = load_iris(filename)
            figures = run_iris(data, labels, args.seed, args.filename is None)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    if args.save:
        results_folder.mkdir(exist_ok=True)
        for name, figure in figures:
            path = results_folder / f"{name}.png"
            figure.savefig(path, dpi=140)
            print("Saved:", path)
    if not args.no_show:
        plt.show()
    for name, figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
