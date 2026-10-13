"""Lab 5: Gaussian classifiers, posterior probabilities, and likelihood ratios."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

LABS_FOLDER = Path(__file__).resolve().parents[1]
for folder in ["lab02", "lab03", "lab04"]:
    sys.path.insert(0, str(LABS_FOLDER / folder))
from lab02 import load_iris, load_fingerprints, IRIS_CLASSES, COLORS
from lab03 import split_data, fit_pca, evaluate_classifier
from lab04 import fit_gaussian, logpdf_GAU_ND, check_reference


MODELS = ["MVG", "TiedMVG", "NaiveBayes"]


# 1. Estimate class means and model-specific covariance matrices.

def fit_classifier(data, labels, model="MVG"):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if labels.ndim != 1 or labels.size != data.shape[1]:
        raise ValueError("There must be one label per sample column.")
    parameters = {}
    counts = {}
    for class_id in np.unique(labels):
        samples = data[:, labels == class_id]
        parameters[int(class_id)] = fit_gaussian(samples)
        counts[int(class_id)] = samples.shape[1]
    if len(parameters) < 2:
        raise ValueError("Classification needs at least two classes.")

    if model == "NaiveBayes":
        for class_id, (mean, covariance) in parameters.items():
            parameters[class_id] = (mean, np.diag(np.diag(covariance)))
    elif model == "TiedMVG":
        shared = np.zeros((data.shape[0], data.shape[0]))
        for class_id, (mean, covariance) in parameters.items():
            shared += counts[class_id] * covariance
        shared /= data.shape[1]
        for class_id, (mean, covariance) in parameters.items():
            parameters[class_id] = (mean, shared.copy())
    return parameters


def class_log_scores(data, parameters):
    classes = np.array(sorted(parameters), dtype=np.int32)
    scores = np.vstack([logpdf_GAU_ND(data, *parameters[int(class_id)])
                        for class_id in classes])
    return scores, classes


# 2. Inference in ordinary probability space and in stable log space.

def class_priors(class_count, priors=None):
    if priors is None:
        return np.full(class_count, 1 / class_count)
    priors = np.asarray(priors, dtype=np.float64)
    if (priors.shape != (class_count,) or not np.isfinite(priors).all() or
            np.any(priors <= 0) or not np.isclose(priors.sum(), 1, rtol=1e-10, atol=1e-12)):
        raise ValueError("Priors must be positive probabilities, one per class, summing to one.")
    return priors


def infer_linear(log_scores, priors=None):
    priors = class_priors(log_scores.shape[0], priors)
    with np.errstate(over="ignore", under="ignore"):
        joint = np.exp(log_scores) * priors[:, None]
        marginal = joint.sum(axis=0)
    if np.any(marginal == 0) or not np.isfinite(marginal).all():
        raise ValueError("Probability-space underflow or overflow: use log-space inference.")
    return {"joint": joint, "marginal": marginal,
            "posterior": joint / marginal[None, :]}


def infer_log(log_scores, priors=None):
    priors = class_priors(log_scores.shape[0], priors)
    log_joint = log_scores + np.log(priors)[:, None]
    log_marginal = logsumexp(log_joint, axis=0)
    log_posterior = log_joint - log_marginal[None, :]
    return {"log_joint": log_joint, "log_marginal": log_marginal,
            "log_posterior": log_posterior, "posterior": np.exp(log_posterior)}


def predict_classes(log_posterior, classes):
    return classes[np.argmax(log_posterior, axis=0)]


def error_rate(predictions, labels):
    if predictions.shape != labels.shape or labels.size == 0:
        raise ValueError("Predictions and labels must have matching, nonempty shapes.")
    return float(np.mean(predictions != labels))


# 3. Binary scores and decisions are separate from parameter estimation.

def compute_llr(data, parameters):
    scores, classes = class_log_scores(data, parameters)
    if classes.size != 2:
        raise ValueError("A log-likelihood ratio needs exactly two classes.")
    return scores[1] - scores[0], classes


def prior_threshold(positive_prior):
    if not 0 < positive_prior < 1:
        raise ValueError("The positive-class prior must be between zero and one.")
    return float(np.log((1 - positive_prior) / positive_prior))


def predict_llr(scores, classes, threshold=0):
    return np.where(scores >= threshold, classes[1], classes[0]).astype(np.int32)


def evaluate_binary(training, validation, model):
    data, labels = training
    parameters = fit_classifier(data, labels, model)
    scores, classes = compute_llr(validation[0], parameters)
    predictions = predict_llr(scores, classes)
    return {"model": model, "parameters": parameters, "scores": scores,
            "predictions": predictions, "errors": int(np.sum(predictions != validation[1])),
            "error_rate": error_rate(predictions, validation[1])}


def correlation_matrix(covariance):
    standard_deviations = np.sqrt(np.diag(covariance))
    if np.any(standard_deviations == 0):
        raise ValueError("Correlation is undefined for a zero-variance feature.")
    return covariance / (standard_deviations[:, None] * standard_deviations[None, :])


def confusion_matrix(predictions, labels, classes):
    positions = {int(class_id): index for index, class_id in enumerate(classes)}
    matrix = np.zeros((classes.size, classes.size), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[positions[int(actual)], positions[int(predicted)]] += 1
    return matrix


def print_binary_results(results, sample_count):
    print(f"{'Model':<15} {'Errors':>10} {'Error rate':>12}")
    for result in results:
        print(f"{result['model']:<15} {result['errors']:>4}/{sample_count:<5} "
              f"{100 * result['error_rate']:11.2f}%")


# 4. Overview plots for Iris and the project comparisons.

def plot_iris_results(results, validation_labels):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    order = np.argsort(validation_labels, kind="stable")
    counts = np.bincount(validation_labels, minlength=3)
    ends = np.cumsum(counts)
    starts = np.concatenate([[0], ends[:-1]])
    centers = (starts + ends - 1) / 2
    for column, result in enumerate(results):
        matrix = confusion_matrix(result["predictions"], validation_labels, result["classes"])
        axis = axes[0, column]
        axis.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(3):
            for col in range(3):
                color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
        axis.set_xticks(range(3), IRIS_CLASSES, rotation=20)
        axis.set_yticks(range(3), IRIS_CLASSES)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_title(f"{result['model']}: {100 * result['error_rate']:.1f}% error")

        axis = axes[1, column]
        image = axis.imshow(result["posterior"][:, order], aspect="auto", cmap="Blues",
                            vmin=0, vmax=1, interpolation="nearest")
        axis.set_xticks(centers, IRIS_CLASSES, rotation=20)
        axis.set_yticks(range(3), IRIS_CLASSES)
        for boundary in ends[:-1]:
            axis.axvline(boundary - 0.5, color="black", linewidth=0.7)
        axis.set_xlabel("Validation samples grouped by actual class")
        axis.set_ylabel("Posterior for class")
        axis.set_title("Class posterior probabilities")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Iris Gaussian classifiers: confusion counts and class probabilities")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_project_covariances(parameters):
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    names = ["Fake", "Genuine"]
    for class_id in [0, 1]:
        covariance = parameters[class_id][1]
        correlation = correlation_matrix(covariance)
        for row, matrix, title in [(0, covariance, "Covariance"),
                                   (1, correlation, "Correlation")]:
            axis = axes[row, class_id]
            bound = 1 if row == 1 else np.abs(matrix).max()
            image = axis.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    color = "white" if abs(matrix[i, j]) > 0.65 * bound else "black"
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                              color=color, fontsize=8)
            axis.set_xticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
            axis.set_yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
            axis.set_xlabel("Feature")
            axis.set_ylabel("Feature")
            axis.set_title(f"{names[class_id]}: {title.lower()}")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Fingerprint training split: covariance and correlation by class")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_project_comparisons(feature_results, pca_results):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    groups = list(feature_results)
    x = np.arange(len(groups))
    for index, model in enumerate(MODELS):
        errors = [100 * feature_results[group][index]["error_rate"] for group in groups]
        bars = axes[0].bar(x + (index - 1) * 0.25, errors, width=0.25,
                           color=COLORS[index], label=model)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
        axes[1].plot(range(1, 7),
                     [100 * row[index]["error_rate"] for row in pca_results],
                     "o-", color=COLORS[index], label=model)
    axes[0].set_xticks(x, groups)
    axes[0].set_title("Original feature subsets")
    axes[0].set_xlabel("Included features")
    axes[0].set_ylim(0, max(100 * item["error_rate"] for row in feature_results.values()
                          for item in row) + 7)
    axes[1].set_title("PCA fitted only on the training split")
    axes[1].set_xlabel("Retained PCA dimensions")
    axes[1].set_xticks(range(1, 7))
    axes[1].set_ylim(0, max(100 * item["error_rate"] for row in pca_results for item in row) + 2)
    for axis in axes:
        axis.set_ylabel("Validation error (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=9)
    figure.suptitle("Fingerprint classifiers: same 4,000/2,000 split, equal priors")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


# 5. Complete Iris and project experiments.

def compare_iris_references(model, linear, logged, folder):
    comparisons = [("SJoint", linear["joint"]), ("Posterior", linear["posterior"]),
                   ("logSJoint", logged["log_joint"]),
                   ("logMarginal", logged["log_marginal"]),
                   ("logPosterior", logged["log_posterior"])]
    checked = 0
    for name, calculated in comparisons:
        filename = folder / f"{name}_{model}.npy"
        if filename.is_file():
            expected = np.load(filename, allow_pickle=False)
            check_reference(calculated, expected, f"{model} {name}")
            checked += 1
    return checked


def run_iris(data, labels, seed, reference_folder):
    training, validation = split_data(data, labels, seed)
    print(f"Iris split: {training[1].size} training, {validation[1].size} validation")
    results = []
    checked = 0
    for model in MODELS:
        parameters = fit_classifier(*training, model)
        log_scores, classes = class_log_scores(validation[0], parameters)
        linear = infer_linear(log_scores)
        logged = infer_log(log_scores)
        difference = float(np.max(np.abs(linear["posterior"] - logged["posterior"])))
        print(f"\n{model}: linear/log posterior difference = {difference:.3e}")
        if not np.allclose(linear["posterior"], logged["posterior"], rtol=0, atol=1e-12):
            raise ValueError("The two inference approaches disagree.")
        # Reference arrays correspond to seed 0, so other splits cannot use them.
        if seed == 0 and reference_folder is not None:
            checked += compare_iris_references(model, linear, logged, reference_folder)
        predictions = predict_classes(logged["log_posterior"], classes)
        results.append({"model": model, "classes": classes, "posterior": logged["posterior"],
                        "predictions": predictions, "parameters": parameters,
                        "errors": int(np.sum(predictions != validation[1])),
                        "error_rate": error_rate(predictions, validation[1])})
    print("\nThree-class Iris results:")
    print_binary_results(results, validation[1].size)

    mask = labels != 0
    binary_training, binary_validation = split_data(data[:, mask], labels[mask], seed)
    binary_results = [evaluate_binary(binary_training, binary_validation, model)
                      for model in ["MVG", "TiedMVG"]]
    print("\nBinary Iris results (Versicolor/Virginica):")
    print_binary_results(binary_results, binary_validation[1].size)
    filename = reference_folder / "llr_MVG.npy" if reference_folder is not None else None
    if seed == 0 and filename is not None and filename.is_file():
        check_reference(binary_results[0]["scores"], np.load(filename, allow_pickle=False),
                        "Binary MVG LLR")
        checked += 1
    lda = evaluate_classifier(binary_training, binary_validation, "LDA")
    same_labels = np.array_equal(binary_results[1]["predictions"], lda["predictions"])
    print("Tied Gaussian and LDA predictions identical:", same_labels)
    if not same_labels:
        raise ValueError("Tied Gaussian and LDA should agree for equal priors.")
    if seed == 0:
        print(f"\nSupplied reference checks completed: {checked}/16")
    else:
        print("\nReference checks skipped: supplied arrays use seed 0.")
    return [("iris_gaussian_classifiers", plot_iris_results(results, validation[1]))]


def run_project(data, labels, seed):
    training, validation = split_data(data, labels, seed)
    print(f"Project split: {training[1].size} training, {validation[1].size} validation")
    feature_results = {}
    for name, indices in [("All 6", list(range(6))), ("1-4", list(range(4))),
                           ("1-2", [0, 1]), ("3-4", [2, 3])]:
        feature_training = (training[0][indices], training[1])
        feature_validation = (validation[0][indices], validation[1])
        results = [evaluate_binary(feature_training, feature_validation, model) for model in MODELS]
        feature_results[name] = results
        print(f"\nFeatures {name}:")
        print_binary_results(results, validation[1].size)

    lda = evaluate_classifier(training, validation, "LDA")
    tied = feature_results["All 6"][1]
    print(f"\nLDA: {lda['errors']}/{validation[1].size} errors ({100 * lda['error_rate']:.2f}%)")
    same_labels = np.array_equal(tied["predictions"], lda["predictions"])
    print("Tied Gaussian and LDA predictions identical:", same_labels)
    if not same_labels:
        raise ValueError("Tied Gaussian and LDA should agree for equal priors.")

    parameters = feature_results["All 6"][0]["parameters"]
    for class_id, name in [(0, "Fake"), (1, "Genuine")]:
        covariance = parameters[class_id][1]
        correlation = correlation_matrix(covariance)
        off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
        print(f"\n{name} covariance:\n{covariance}")
        print(f"{name} correlation:\n{correlation}")
        print("Largest absolute off-diagonal correlation:", np.abs(off_diagonal).max())

    pca_results = []
    for dimensions in range(1, data.shape[0] + 1):
        projection, _ = fit_pca(training[0], dimensions)
        transformed_training = (projection.T @ training[0], training[1])
        transformed_validation = (projection.T @ validation[0], validation[1])
        results = [evaluate_binary(transformed_training, transformed_validation, model)
                   for model in MODELS]
        pca_results.append(results)
        print(f"\nPCA dimensions: {dimensions}")
        print_binary_results(results, validation[1].size)

    print("\nAll comparisons use equal priors and LLR threshold zero.")
    print("Feature/PCA choices are validation comparisons, not independent test results.")
    print("The separate evaluation dataset remains unused.")
    return [("fingerprints_covariance", plot_project_covariances(parameters)),
            ("fingerprints_model_comparison", plot_project_comparisons(feature_results, pca_results))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="optional custom Iris CSV or project file")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    np.set_printoptions(precision=6, suppress=True)
    lab_folder = Path(__file__).resolve().parent
    default_name = "traindata.txt" if args.project else "iris.csv"
    filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data" / default_name
    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            figures = run_project(data, labels, args.seed)
        else:
            data, labels = load_iris(filename)
            # Custom files may not match the supplied Iris references.
            references = lab_folder / "data" if args.filename is None else None
            figures = run_iris(data, labels, args.seed, references)
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
