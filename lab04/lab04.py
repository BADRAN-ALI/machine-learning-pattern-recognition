"""Lab 4: Gaussian densities, maximum-likelihood estimates, and feature fits."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

LABS_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS_FOLDER / "lab02"))
from lab02 import load_fingerprints, COLORS


# 1. Gaussian log-density for one sample or a matrix of sample columns.

def logpdf_GAU_ND(samples, mean, covariance):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
    covariance = np.asarray(covariance, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("Samples must be a feature-by-sample matrix.")
    features = samples.shape[0]
    if mean.shape != (features, 1) or covariance.shape != (features, features):
        raise ValueError("Mean and covariance shapes must match the number of features.")
    if not (np.isfinite(samples).all() and np.isfinite(mean).all() and
            np.isfinite(covariance).all()):
        raise ValueError("Samples, mean, and covariance must contain finite numbers.")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The covariance must be symmetric.")

    # Cholesky checks positive definiteness and avoids forming an inverse.
    factor = np.linalg.cholesky(covariance)
    centered = samples - mean
    transformed = np.linalg.solve(factor, centered)
    squared_distance = (transformed ** 2).sum(axis=0)
    log_determinant = np.linalg.slogdet(covariance)[1]
    return -0.5 * (features * np.log(2 * np.pi) + log_determinant + squared_distance)


# 2. Maximum-likelihood parameter estimates and log-likelihood.

def fit_gaussian(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("Gaussian estimation needs a nonempty feature-by-sample matrix.")
    if not np.isfinite(samples).all():
        raise ValueError("Samples must contain finite numbers.")
    mean = samples.mean(axis=1, keepdims=True)
    centered = samples - mean
    covariance = centered @ centered.T / samples.shape[1]
    return mean, covariance


def loglikelihood(samples, mean, covariance):
    return float(logpdf_GAU_ND(samples, mean, covariance).sum())


def check_reference(calculated, expected, name):
    if calculated.shape != expected.shape:
        raise ValueError(f"{name}: the reference and calculated shapes differ.")
    difference = float(np.max(np.abs(calculated - expected)))
    print(f"{name}: maximum absolute difference = {difference:.3e}")
    if not np.allclose(calculated, expected, rtol=0, atol=1e-10):
        raise ValueError(f"{name}: the supplied reference check failed.")
    return difference


def print_estimates(samples, name):
    mean, covariance = fit_gaussian(samples)
    print(f"\n{name}: {samples.shape[1]} samples, {samples.shape[0]} features")
    print("ML mean:\n", mean)
    print("ML covariance:\n", covariance)
    print("Log-likelihood:", loglikelihood(samples, mean, covariance))
    return mean, covariance


def compare_1d_parameters(samples, mean, covariance):
    standard_deviation = np.sqrt(covariance[0, 0])
    best_likelihood = loglikelihood(samples, mean, covariance)
    print("\nChanging the 1D parameters (difference from the ML log-likelihood):")
    print(f"{'Mean shift / std.':>18} {'Variance factor':>18} {'Log-likelihood change':>24}")
    for shift in [-1, 0, 1]:
        for scale in [0.5, 1, 2]:
            trial_mean = mean + shift * standard_deviation
            trial_covariance = covariance * scale
            likelihood = loglikelihood(samples, trial_mean, trial_covariance)
            print(f"{shift:18.1f} {scale:18.1f} {likelihood - best_likelihood:24.4f}")


# 3. One overview for the supplied Gaussian datasets.

def plot_gaussian_examples(samples_1d, samples_nd, mean_1d, covariance_1d,
                           mean_nd, covariance_nd, bins):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    grid = np.linspace(-8, 12, 1000).reshape(1, -1)

    axis = axes[0, 0]
    density = np.exp(logpdf_GAU_ND(grid, np.array([[1.]]), np.array([[2.]])))
    axis.plot(grid.ravel(), density, color=COLORS[0])
    axis.set_title("Example Gaussian: mean 1, variance 2")
    axis.set_xlabel("Value")
    axis.set_ylabel("Density")

    axis = axes[0, 1]
    axis.hist(samples_1d.ravel(), bins=bins, density=True, alpha=0.5,
              color=COLORS[0], label="Samples")
    density = np.exp(logpdf_GAU_ND(grid, mean_1d, covariance_1d))
    axis.plot(grid.ravel(), density, color="black", linewidth=2, label="ML Gaussian")
    axis.set_title("X1D: estimated density and sample histogram")
    axis.set_xlabel("Value")
    axis.set_ylabel("Density")
    axis.legend(fontsize=9)

    axis = axes[1, 0]
    axis.scatter(samples_nd[0], samples_nd[1], s=20, alpha=0.7, color=COLORS[0])
    x = np.linspace(samples_nd[0].min() - 1, samples_nd[0].max() + 1, 120)
    y = np.linspace(samples_nd[1].min() - 1, samples_nd[1].max() + 1, 120)
    x_grid, y_grid = np.meshgrid(x, y)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    # The first two features form the 2D marginal when more features are supplied.
    density = np.exp(logpdf_GAU_ND(positions, mean_nd[:2], covariance_nd[:2, :2]))
    contours = axis.contour(x_grid, y_grid, density.reshape(x_grid.shape),
                            levels=5, cmap="Greys")
    axis.clabel(contours, fontsize=8)
    axis.set_title("XND: samples and estimated density contours")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")

    axis = axes[1, 1]
    standard_deviation = np.sqrt(covariance_1d[0, 0])
    means = np.linspace(mean_1d[0, 0] - 1.5 * standard_deviation,
                        mean_1d[0, 0] + 1.5 * standard_deviation, 101)
    likelihoods = [loglikelihood(samples_1d, np.array([[value]]), covariance_1d)
                   for value in means]
    axis.plot(means, likelihoods, color=COLORS[0])
    axis.axvline(mean_1d[0, 0], color="black", linestyle="--", label="ML mean")
    axis.set_title("X1D: change the mean, keep ML variance fixed")
    axis.set_xlabel("Candidate mean")
    axis.set_ylabel("Log-likelihood")
    axis.legend(fontsize=9)

    for axis in axes.ravel():
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("Gaussian density and maximum-likelihood estimation")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def run_gaussian_examples(data_folder, bins):
    arrays = {name: np.load(data_folder / f"{name}.npy", allow_pickle=False)
              for name in ["X1D", "XND", "muND", "CND", "llGAU", "llND"]}
    grid = np.linspace(-8, 12, 1000).reshape(1, -1)
    reference_1d = logpdf_GAU_ND(grid, np.array([[1.]]), np.array([[2.]]))
    reference_nd = logpdf_GAU_ND(arrays["XND"], arrays["muND"], arrays["CND"])
    check_reference(reference_1d, arrays["llGAU"], "1D log-density")
    check_reference(reference_nd, arrays["llND"], "ND log-density")
    print("Both supplied reference checks passed.")

    mean_nd, covariance_nd = print_estimates(arrays["XND"], "XND")
    mean_1d, covariance_1d = print_estimates(arrays["X1D"], "X1D")
    compare_1d_parameters(arrays["X1D"], mean_1d, covariance_1d)
    figure = plot_gaussian_examples(arrays["X1D"], arrays["XND"], mean_1d,
                                    covariance_1d, mean_nd, covariance_nd, bins)
    return "gaussian_estimation", figure


# 4. Project: fit a separate univariate Gaussian for each class and feature.

def fit_project_features(data, labels):
    estimates = {}
    for class_id in np.unique(labels):
        means = []
        variances = []
        for feature in range(data.shape[0]):
            samples = data[feature:feature + 1, labels == class_id]
            mean, covariance = fit_gaussian(samples)
            means.append(mean[0, 0])
            variances.append(covariance[0, 0])
        estimates[int(class_id)] = (np.array(means), np.array(variances))
    return estimates


def plot_project_fits(data, labels, estimates, bins):
    figure, axes = plt.subplots(2, data.shape[0], figsize=(18, 7))
    classes = ["Fake", "Genuine"]
    for class_id in [0, 1]:
        means, variances = estimates[class_id]
        for feature in range(data.shape[0]):
            axis = axes[class_id, feature]
            samples = data[feature, labels == class_id]
            axis.hist(samples, bins=bins, density=True, alpha=0.5,
                      color=COLORS[class_id], label="Samples")
            grid = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 400)
            density = np.exp(logpdf_GAU_ND(grid.reshape(1, -1),
                             np.array([[means[feature]]]), np.array([[variances[feature]]])))
            axis.plot(grid, density, color="black", linewidth=1.8, label="ML Gaussian")
            axis.set_title(f"{classes[class_id]}: feature {feature + 1}", fontsize=10)
            axis.set_xlabel("Value", fontsize=9)
            if feature == 0:
                axis.set_ylabel("Density", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=0.15)
            axis.set_axisbelow(True)
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 0.96), frameon=False)
    figure.suptitle("Fingerprint features: a separate Gaussian for each class and feature")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def run_project(filename, bins):
    data, labels = load_fingerprints(filename)
    estimates = fit_project_features(data, labels)
    print("Project samples:", data.shape[1])
    print(f"{'Feature':<9} {'Fake mean':>12} {'Fake var.':>12} "
          f"{'Genuine mean':>14} {'Genuine var.':>14}")
    for feature in range(data.shape[0]):
        fake_mean, fake_variance = estimates[0]
        genuine_mean, genuine_variance = estimates[1]
        print(f"{feature + 1:<9} {fake_mean[feature]:12.4f} {fake_variance[feature]:12.4f} "
              f"{genuine_mean[feature]:14.4f} {genuine_variance[feature]:14.4f}")
    print("\nThis is qualitative fitting on the full training file, not classifier evaluation.")
    return "fingerprints_gaussian_fits", plot_project_fits(data, labels, estimates, bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="optional project training CSV")
    parser.add_argument("--project", action="store_true", help="fit fingerprint feature densities")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--save", action="store_true", help="save the overview PNG")
    parser.add_argument("--no-show", action="store_true", help="run without plot windows")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("The number of bins must be positive.")
    if args.filename and not args.project:
        parser.error("A custom filename is only used with --project.")
    np.set_printoptions(precision=8, suppress=True)
    lab_folder = Path(__file__).resolve().parent
    try:
        if args.project:
            filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data/traindata.txt"
            name, figure = run_project(filename, args.bins)
        else:
            name, figure = run_gaussian_examples(lab_folder / "data", args.bins)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    if args.save:
        output_folder = lab_folder / "results"
        output_folder.mkdir(exist_ok=True)
        path = output_folder / f"{name}.png"
        figure.savefig(path, dpi=140)
        print("Saved:", path)
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
