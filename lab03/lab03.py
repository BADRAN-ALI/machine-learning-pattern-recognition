"""Lab 3: PCA, LDA, and classification with a training/validation split."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

# Reuse Lab 2's loaders and plotting function without copying the data files.
LABS_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS_FOLDER / "lab02"))
from lab02 import load_iris, load_fingerprints, plot_feature_grid, IRIS_CLASSES, COLORS


# 1. Principal Component Analysis.

def vcol(vector):
    return vector.reshape(-1, 1)


def vrow(vector):
    return vector.reshape(1, -1)


def fix_signs(matrix):
    """Choose consistent signs for plots; eigenvector signs are arbitrary."""
    result = matrix.copy()
    for column in range(result.shape[1]):
        largest = np.argmax(np.abs(result[:, column]))
        if result[largest, column] < 0:
            result[:, column] *= -1
    return result


def fit_pca(data, dimensions, method="eigh"):
    if not 1 <= dimensions <= data.shape[0]:
        raise ValueError("PCA dimensions must be between 1 and the number of features.")
    centered = data - data.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / data.shape[1]
    if method == "eigh":
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
    elif method == "svd":
        vectors, values, _ = np.linalg.svd(covariance)
    else:
        raise ValueError("PCA method must be 'eigh' or 'svd'.")
    return fix_signs(vectors[:, :dimensions]), values


# 2. Linear Discriminant Analysis, with two equivalent methods.

def class_scatter(data, labels):
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("LDA needs at least two classes.")
    overall_mean = data.mean(axis=1, keepdims=True)
    between = np.zeros((data.shape[0], data.shape[0]))
    within = np.zeros_like(between)
    for class_id in classes:
        samples = data[:, labels == class_id]
        class_mean = samples.mean(axis=1, keepdims=True)
        difference = class_mean - overall_mean
        between += samples.shape[1] * (difference @ difference.T)
        centered = samples - class_mean
        within += centered @ centered.T
    return between / data.shape[1], within / data.shape[1]


def fit_lda(data, labels, dimensions, method="generalized"):
    maximum = min(data.shape[0], np.unique(labels).size - 1)
    if not 1 <= dimensions <= maximum:
        raise ValueError(f"LDA can use between 1 and {maximum} directions here.")
    between, within = class_scatter(data, labels)
    if method == "generalized":
        values, vectors = eigh(between, within)
    elif method == "whitening":
        variances, axes = np.linalg.eigh(within)
        if np.any(variances <= 0):
            raise ValueError("The within-class covariance must be positive definite.")
        whitening = axes @ np.diag(1 / np.sqrt(variances)) @ axes.T
        transformed = whitening @ between @ whitening.T
        values, directions = np.linalg.eigh(transformed)
        vectors = whitening.T @ directions
    else:
        raise ValueError("LDA method must be 'generalized' or 'whitening'.")
    order = np.argsort(values)[::-1]
    return fix_signs(vectors[:, order[:dimensions]]), values[order]


def orthonormal_basis(matrix):
    """A basis for comparing subspaces, not for changing the LDA classifier."""
    vectors, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return vectors


def subspace_difference(first, second):
    first_basis = orthonormal_basis(first)
    second_basis = orthonormal_basis(second)
    return np.linalg.norm(first_basis @ first_basis.T - second_basis @ second_basis.T)


# 3. Split first, then estimate projections and threshold from training data.

def split_data(data, labels, seed=0):
    # Local legacy RNG reproduces the sheet's seed-0 split without changing global state.
    indices = np.random.RandomState(seed).permutation(data.shape[1])
    training_count = (2 * data.shape[1]) // 3
    training = indices[:training_count]
    validation = indices[training_count:]
    classes = np.unique(labels)
    if (np.unique(labels[training]).size != classes.size or
            np.unique(labels[validation]).size != classes.size):
        raise ValueError("Both splits must contain every class; try another seed.")
    return (data[:, training], labels[training]), (data[:, validation], labels[validation])


def fit_binary_classifier(data, labels, method="lda", pca_dimensions=None):
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError("This classifier needs exactly two classes.")
    if method == "pca":
        direction, _ = fit_pca(data, 1)
    elif method == "lda":
        if pca_dimensions is None:
            direction, _ = fit_lda(data, labels, 1)
        else:
            pca, _ = fit_pca(data, pca_dimensions)
            lda, _ = fit_lda(pca.T @ data, labels, 1)
            direction = pca @ lda
    else:
        raise ValueError("Binary method must be 'pca' or 'lda'.")

    projected = (direction.T @ data).ravel()
    # The higher label must have the higher projected training mean.
    if projected[labels == classes[1]].mean() < projected[labels == classes[0]].mean():
        direction = -direction
        projected = -projected
    threshold = (projected[labels == classes[0]].mean() +
                 projected[labels == classes[1]].mean()) / 2
    return direction, threshold, classes


def predict(scores, threshold, classes):
    return np.where(scores >= threshold, classes[1], classes[0]).astype(np.int32)


def evaluate_classifier(training, validation, name, method="lda", pca_dimensions=None):
    train_data, train_labels = training
    validation_data, validation_labels = validation
    direction, threshold, classes = fit_binary_classifier(train_data, train_labels,
                                                         method, pca_dimensions)
    training_scores = (direction.T @ train_data).ravel()
    validation_scores = (direction.T @ validation_data).ravel()
    predictions = predict(validation_scores, threshold, classes)
    errors = int(np.sum(predictions != validation_labels))
    return {"name": name, "direction": direction, "threshold": threshold,
            "classes": classes, "training_scores": training_scores,
            "validation_scores": validation_scores, "predictions": predictions,
            "errors": errors, "error_rate": errors / validation_labels.size}


def tune_threshold(scores, labels, classes, baseline):
    """Search validation cutoffs; the resulting error is a tuning result."""
    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2
    thresholds = np.sort(np.append(midpoints, [baseline,
                         np.nextafter(unique_scores[0], -np.inf),
                         np.nextafter(unique_scores[-1], np.inf)]))
    error_counts = np.array([np.sum(predict(scores, value, classes) != labels)
                             for value in thresholds])
    candidates = np.flatnonzero(error_counts == error_counts.min())
    best = candidates[np.argmin(np.abs(thresholds[candidates] - baseline))]
    return thresholds[best], int(error_counts[best]), thresholds, error_counts


def print_results(results, validation_labels):
    print(f"\n{'Method':<17} {'Errors':>9} {'Error rate':>12} {'Threshold':>12}")
    for result in results:
        print(f"{result['name']:<17} {result['errors']:>4}/{validation_labels.size:<4} "
              f"{100 * result['error_rate']:11.2f}% {result['threshold']:12.4f}")


# 4. Plot comparisons without generating one file per feature pair.

def draw_histogram(axis, scores, labels, names, bins, title, threshold=None):
    edges = np.histogram_bin_edges(scores, bins=bins)
    for class_id in np.unique(labels):
        axis.hist(scores[labels == class_id], bins=edges, density=True, alpha=0.5,
                  color=COLORS[class_id], label=names[class_id])
    if threshold is not None:
        axis.axvline(threshold, color="black", linestyle="--", label="Training midpoint")
    axis.set_title(title)
    axis.set_xlabel("Projected value")
    axis.set_ylabel("Density")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.15)


def plot_iris_comparison(data, labels, pca, lda, bins):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, projection, title in [(axes[0, 0], pca, "PCA: first two directions"),
                                     (axes[0, 1], lda, "LDA: two discriminant directions")]:
        projected = projection.T @ data
        for class_id in np.unique(labels):
            mask = labels == class_id
            axis.scatter(projected[0, mask], projected[1, mask], s=20, alpha=0.75,
                         color=COLORS[class_id], label=IRIS_CLASSES[class_id])
        axis.set_title(title)
        axis.set_xlabel("Direction 1")
        axis.set_ylabel("Direction 2")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    draw_histogram(axes[1, 0], (pca[:, :1].T @ data).ravel(), labels,
                   IRIS_CLASSES, bins, "PCA: first direction")
    draw_histogram(axes[1, 1], (lda[:, :1].T @ data).ravel(), labels,
                   IRIS_CLASSES, bins, "LDA: first direction")
    figure.suptitle("Iris: preserve variance with PCA, use class labels with LDA")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_binary_comparison(results, training_labels, validation_labels):
    figure, axes = plt.subplots(len(results), 2, figsize=(11, 4 * len(results)))
    names = {1: "Versicolor", 2: "Virginica"}
    for row, result in enumerate(results):
        draw_histogram(axes[row, 0], result["training_scores"], training_labels, names,
                       5, f"{result['name']}: training", result["threshold"])
        draw_histogram(axes[row, 1], result["validation_scores"], validation_labels, names,
                       5, f"Validation: {result['errors']} errors", result["threshold"])
    figure.suptitle("Binary Iris: directions and thresholds fitted on training samples")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def run_iris(data, labels, seed, bins):
    pca, values = fit_pca(data, 2)
    pca_svd, _ = fit_pca(data, 2, "svd")
    lda, lda_values = fit_lda(data, labels, 2)
    lda_whitening, _ = fit_lda(data, labels, 2, "whitening")
    between, within = class_scatter(data, labels)
    print("Dataset mean:", data.mean(axis=1))
    centered = data - data.mean(axis=1, keepdims=True)
    print("Dataset covariance:\n", centered @ centered.T / data.shape[1])
    print("PCA eigenvalues:", values)
    print("PCA retained variance (2 dimensions):", values[:2].sum() / values.sum())
    print("LDA eigenvalues:", lda_values)
    print("Between-class covariance:\n", between)
    print("Within-class covariance:\n", within)
    print("PCA eigh/SVD subspace difference:", subspace_difference(pca, pca_svd))
    print("LDA generalized/whitening subspace difference:", subspace_difference(lda, lda_whitening))

    mask = labels != 0
    training, validation = split_data(data[:, mask], labels[mask], seed)
    print(f"\nBinary Iris split: {training[1].size} training, {validation[1].size} validation")
    results = [evaluate_classifier(training, validation, "LDA"),
               evaluate_classifier(training, validation, "PCA only", method="pca")]
    for dimensions in [2, 3]:
        results.append(evaluate_classifier(training, validation, f"PCA {dimensions} + LDA",
                                           pca_dimensions=dimensions))
    print_results(results, validation[1])
    print("Projection uses the original data, as in the sheet; centering would shift the axes.")
    return [("iris_pca_lda", plot_iris_comparison(data, labels, pca, lda, bins)),
            ("iris_binary", plot_binary_comparison(results[:2], training[1], validation[1]))]


def plot_project_classification(full_scores, full_labels, training, validation,
                                baseline, results, best, thresholds, errors, bins):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    names = ["Fake", "Genuine"]
    draw_histogram(axes[0, 0], full_scores, full_labels, names, bins, "Full-data LDA: descriptive plot")
    draw_histogram(axes[0, 1], baseline["training_scores"], training[1], names, bins,
                   "Classifier: training", baseline["threshold"])
    draw_histogram(axes[0, 2], baseline["validation_scores"], validation[1], names, bins,
                   "Classifier: validation", baseline["threshold"])
    axis = axes[1, 0]
    axis.plot(thresholds, 100 * errors / validation[1].size, color=COLORS[0])
    axis.axvline(baseline["threshold"], color="black", linestyle="--", label="Training midpoint")
    axis.axvline(best, color=COLORS[1], linestyle=":", label="Selected on validation")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Validation error (%)")
    axis.set_title("Threshold search: tuning on validation labels")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    dimensions = np.arange(1, len(results) + 1)
    axis.plot(dimensions, [100 * item["error_rate"] for item in results], "o-", color=COLORS[0])
    axis.axhline(100 * baseline["error_rate"], color="black", linestyle="--", label="LDA without PCA")
    axis.set_xticks(dimensions)
    axis.set_xlabel("PCA dimensions")
    axis.set_ylabel("Validation error (%)")
    axis.set_title("PCA + LDA: training midpoint thresholds")
    axis.set_ylim(0, max(100 * item["error_rate"] for item in results) + 2)
    for dimension, result in zip(dimensions, results):
        axis.annotate(f"{100 * result['error_rate']:.2f}%",
                      (dimension, 100 * result["error_rate"]),
                      xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    negative, positive = baseline["classes"]
    predicted = baseline["predictions"]
    false_positive = int(np.sum((validation[1] == negative) & (predicted == positive)))
    false_negative = int(np.sum((validation[1] == positive) & (predicted == negative)))
    axis.bar(["Fake as genuine", "Genuine as fake"], [false_positive, false_negative],
             color=[COLORS[0], COLORS[1]])
    axis.set_ylabel("Misclassified samples")
    axis.set_title("LDA baseline: error counts")
    for axis in axes.ravel():
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("Fingerprint classification: training estimates, validation comparisons")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def run_project(data, labels, seed, bins):
    pca, eigenvalues = fit_pca(data, data.shape[0])
    projected = pca.T @ data
    print("PCA eigenvalues:", eigenvalues)
    print("Cumulative retained variance:", np.cumsum(eigenvalues) / eigenvalues.sum())
    print("Full PCA reconstruction error:", np.max(np.abs(pca @ projected - data)))
    full_lda, _ = fit_lda(data, labels, 1)
    full_scores = (full_lda.T @ data).ravel()
    if full_scores[labels == 1].mean() < full_scores[labels == 0].mean():
        full_scores = -full_scores

    training, validation = split_data(data, labels, seed)
    print(f"\nProject split: {training[1].size} training, {validation[1].size} validation")
    baseline = evaluate_classifier(training, validation, "LDA")
    results = [evaluate_classifier(training, validation, f"PCA {dimensions} + LDA",
                                   pca_dimensions=dimensions)
               for dimensions in range(1, data.shape[0] + 1)]
    print_results([baseline, *results], validation[1])
    best, best_errors, thresholds, errors = tune_threshold(baseline["validation_scores"],
                                                         validation[1], baseline["classes"],
                                                         baseline["threshold"])
    print(f"\nValidation-selected threshold: {best:.4f}")
    print(f"Validation tuning result: {best_errors}/{validation[1].size} errors "
          f"({100 * best_errors / validation[1].size:.2f}%)")
    print("This uses validation labels and is not an independent test result.")
    print("The separate evaluation dataset is not used in this lab.")

    names = [f"PC {index}" for index in range(1, data.shape[0] + 1)]
    return [("fingerprints_pca", plot_feature_grid(projected, labels, names,
             ["Fake", "Genuine"], bins, "Fingerprint features after full PCA: an orthogonal rotation")),
            ("fingerprints_classification", plot_project_classification(full_scores, labels,
             training, validation, baseline, results, best, thresholds, errors, bins))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?")
    parser.add_argument("--project", action="store_true", help="run the fingerprint project")
    parser.add_argument("--seed", type=int, default=0, help="training/validation split seed")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--save", action="store_true", help="save the two overview PNGs")
    parser.add_argument("--no-show", action="store_true", help="run without plot windows")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("The number of bins must be positive.")
    np.set_printoptions(precision=6, suppress=True)
    default_name = "traindata.txt" if args.project else "iris.csv"
    filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data" / default_name
    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            figures = run_project(data, labels, args.seed, args.bins)
        else:
            data, labels = load_iris(filename)
            figures = run_iris(data, labels, args.seed, args.bins)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))

    if args.save:
        folder = Path(__file__).resolve().parent / "results"
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
