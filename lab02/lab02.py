"""Lab 2: load data, visualize features, and calculate descriptive statistics."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IRIS_FEATURES = ["Sepal length (cm)", "Sepal width (cm)",
                 "Petal length (cm)", "Petal width (cm)"]
IRIS_CLASSES = ["Setosa", "Versicolor", "Virginica"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]


# 1. Load the data: rows are features, columns are samples.

def mcol(vector):
    return vector.reshape(-1, 1)


def mrow(vector):
    return vector.reshape(1, -1)


def load_iris(filename):
    samples = []
    labels = []
    class_ids = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    with open(filename, newline="", encoding="utf-8") as file:
        for line_number, fields in enumerate(csv.reader(file), 1):
            if not fields or all(not value.strip() for value in fields):
                continue
            if len(fields) != 5:
                raise ValueError(f"Line {line_number}: expected four features and a class.")
            class_name = fields[4].strip()
            if class_name not in class_ids:
                raise ValueError(f"Line {line_number}: unknown class {class_name}.")
            samples.append([float(value) for value in fields[:4]])
            labels.append(class_ids[class_name])

    if not samples:
        raise ValueError("The file contains no samples.")
    data = np.array(samples, dtype=np.float64).T
    if not np.isfinite(data).all():
        raise ValueError("All feature values must be finite numbers.")
    return data, np.array(labels, dtype=np.int32)


# 2. Mean, centering, covariance, variance, and standard deviation.

def calculate_statistics(data):
    mean = mcol(data.mean(axis=1))
    centered = data - mean
    # The lab uses the empirical covariance with denominator N, not N - 1.
    covariance = centered @ centered.T / data.shape[1]
    variance = data.var(axis=1)
    standard_deviation = data.std(axis=1)
    return mean, centered, covariance, variance, standard_deviation


def print_statistics(data, title, feature_names):
    mean, centered, covariance, variance, standard_deviation = calculate_statistics(data)
    print(f"\n{title} ({data.shape[1]} samples)")
    print(f"{'Feature':<22} {'Mean':>9} {'Variance':>10} {'Std. dev.':>10}")
    for index, name in enumerate(feature_names):
        print(f"{name:<22} {mean[index, 0]:9.4f} {variance[index]:10.4f} "
              f"{standard_deviation[index]:10.4f}")
    print("Covariance matrix:")
    print(covariance)


def print_dataset_statistics(data, labels, feature_names, class_names):
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print_statistics(data, "Whole dataset", feature_names)
    for class_id in np.unique(labels):
        class_data = data[:, labels == class_id]
        print_statistics(class_data, class_names[class_id], feature_names)

    mean, centered, covariance, variance, standard_deviation = calculate_statistics(data)
    print("\nCentered feature means (approximately zero):")
    print(centered.mean(axis=1))


# 3. One figure contains every histogram and ordered feature pair.

def plot_feature_grid(data, labels, feature_names, class_names, bins, title):
    feature_count = data.shape[0]
    figure, axes = plt.subplots(feature_count, feature_count,
                                figsize=(3 * feature_count, 3 * feature_count))

    for row in range(feature_count):
        for column in range(feature_count):
            axis = axes[row, column]
            if row == column:
                # Shared bin boundaries make the class densities easier to compare.
                edges = np.histogram_bin_edges(data[row], bins=bins)
                for class_id in np.unique(labels):
                    values = data[row, labels == class_id]
                    axis.hist(values, bins=edges, density=True, alpha=0.5,
                              color=COLORS[class_id], label=class_names[class_id])
                axis.set_title(feature_names[row], fontsize=10)
                axis.set_ylabel("Density", fontsize=9)
            else:
                for class_id in np.unique(labels):
                    mask = labels == class_id
                    axis.scatter(data[column, mask], data[row, mask], s=12,
                                 alpha=0.65, color=COLORS[class_id])
                if column == 0:
                    axis.set_ylabel(feature_names[row], fontsize=9)

            if row == feature_count - 1:
                axis.set_xlabel(feature_names[column], fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=0.15)
            axis.set_axisbelow(True)

    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", ncol=len(names),
                  bbox_to_anchor=(0.5, 0.96), frameon=False)
    figure.suptitle(title, fontsize=16, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


# 4. Project: fingerprint spoofing data, with six features and labels 0/1.

def load_fingerprints(filename):
    values = np.loadtxt(filename, delimiter=",", ndmin=2)
    if values.shape[0] == 0 or values.shape[1] != 7:
        raise ValueError("Project rows must contain six features and one label.")
    if not np.isfinite(values).all():
        raise ValueError("All project values must be finite numbers.")
    if not np.isin(values[:, -1], [0, 1]).all():
        raise ValueError("Project labels must be 0 (fake) or 1 (genuine).")
    if len(np.unique(values[:, -1])) != 2:
        raise ValueError("Both fake and genuine samples are needed for the comparison.")
    return values[:, :6].T.copy(), values[:, -1].astype(np.int32)


def print_project_comparison(data, labels):
    fake = data[:, labels == 0]
    genuine = data[:, labels == 1]
    print("\nProject: compare the requested feature groups")
    for first in [0, 2, 4]:
        print(f"\nFeatures {first + 1} and {first + 2}:")
        for feature in [first, first + 1]:
            fake_values = fake[feature]
            genuine_values = genuine[feature]
            low = max(fake_values.min(), genuine_values.min())
            high = min(fake_values.max(), genuine_values.max())
            print(f"Feature {feature + 1}:")
            print(f"  Fake: mean {fake_values.mean():.4f}, variance {fake_values.var():.4f}")
            print(f"  Genuine: mean {genuine_values.mean():.4f}, variance {genuine_values.var():.4f}")
            if low <= high:
                print(f"  Intersection of observed ranges: [{low:.4f}, {high:.4f}]")
            else:
                print("  Observed ranges do not intersect.")
    print("\nRange intersections alone do not measure distribution overlap.")
    print("Use the histograms to assess overlap and visible peaks.")
    print("Use the feature 5/6 scatter plot to inspect clusters within each class.")
    print("Repeat with different --bins values before drawing conclusions about modes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="Iris CSV or project training file")
    parser.add_argument("--project", action="store_true", help="analyze fingerprint data")
    parser.add_argument("--bins", type=int, default=10, help="histogram bins, default 10")
    parser.add_argument("--save", action="store_true", help="save overview PNGs")
    parser.add_argument("--no-show", action="store_true", help="run without plot windows")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("The number of bins must be positive.")

    lab_folder = Path(__file__).resolve().parent
    if args.project and args.filename is None:
        parser.error("Pass the path to trainData.txt when using --project.")
    filename = Path(args.filename) if args.filename else lab_folder / "data/iris.csv"
    np.set_printoptions(precision=4, suppress=True)

    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            features = [f"Feature {index}" for index in range(1, 7)]
            classes = ["Fake", "Genuine"]
            prefix = "fingerprints"
            title = "Fingerprint spoofing: feature distributions"
        else:
            data, labels = load_iris(filename)
            features = IRIS_FEATURES
            classes = IRIS_CLASSES
            prefix = "iris"
            title = "Iris: feature distributions and class overlap"
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print_dataset_statistics(data, labels, features, classes)
    if args.project:
        print_project_comparison(data, labels)

    mean, centered, covariance, variance, standard_deviation = calculate_statistics(data)
    figures = [plot_feature_grid(data, labels, features, classes, args.bins, title)]
    if not args.project:
        figures.append(plot_feature_grid(centered, labels, features, classes, args.bins,
                                         "Iris: centered by the whole-dataset mean"))
    if args.save:
        output_folder = lab_folder / "results"
        output_folder.mkdir(exist_ok=True)
        for figure, suffix in zip(figures, ["overview", "centered"]):
            path = output_folder / f"{prefix}_{suffix}.png"
            figure.savefig(path, dpi=140)
            print("Saved:", path)

    if not args.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
