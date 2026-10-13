"""Lab 8: linear and kernel SVMs, dual optimization, and fingerprint experiments."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

LABS_FOLDER = Path(__file__).resolve().parents[1]
for folder in ["lab02", "lab03", "lab05", "lab06"]:
    sys.path.insert(0, str(LABS_FOLDER / folder))
from lab02 import load_iris, load_fingerprints, COLORS
from lab03 import split_data
from lab06 import evaluate_scores

LINEAR_CS = np.logspace(-5, 0, 11)
RBF_CS = np.logspace(-3, 2, 11)
RBF_GAMMAS = np.exp(np.array([-4., -3., -2., -1.]))
PROJECT_GAP_TOLERANCE = 1e-4


# 1. The dual objective and box-constrained optimization.

def check_training(data, labels, penalty):
    if (data.ndim != 2 or min(data.shape) == 0 or labels.shape != (data.shape[1],) or
            not np.isfinite(data).all() or not np.array_equal(np.unique(labels), [0, 1])):
        raise ValueError("Use finite sample columns with one binary label per sample and both classes.")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("C must be finite and positive.")


def make_dual_objective(hessian):
    def objective(alpha):
        product = hessian @ alpha
        value = 0.5 * (alpha @ product) - alpha.sum()
        return float(value), product - 1
    return objective


def solve_dual(objective, samples, penalty, initial=None, tolerance=1e-5, factr=np.nan):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("The optimizer tolerance must be finite and positive.")
    start = np.zeros(samples) if initial is None else np.clip(initial, 0, penalty)
    if start.shape != (samples,) or not np.isfinite(start).all():
        raise ValueError("The initial alpha vector must have one element per training sample.")
    def gap_at(alpha):
        value, gradient = objective(alpha)
        norm = alpha @ (gradient + 1)
        primal = 0.5 * norm + penalty * np.maximum(0, -gradient).sum()
        return primal + value, primal

    # On the larger project, a small primal-dual gap can certify convergence
    # before the projected gradient reaches an unnecessarily strict tolerance.
    project = np.isfinite(factr)
    for attempt in range(6):
        iterations = [0]

        def check_gap(alpha):
            iterations[0] += 1
            if iterations[0] % 10 == 0:
                gap, primal = gap_at(alpha)
                if gap <= PROJECT_GAP_TOLERANCE * max(primal, 1e-6):
                    raise StopIteration

        alpha, value, info = fmin_l_bfgs_b(
            objective, start, bounds=[(0, penalty)] * samples,
            # Tiny boxes must not count as convergence at alpha = 0.
            factr=factr, pgtol=min(tolerance, penalty * 1e-3),
            maxiter=200000, maxfun=200000, m=30 if project else 10,
            maxls=100 if project else 20, callback=check_gap if project else None)
        gap, primal = gap_at(alpha)
        if (project and gap <= PROJECT_GAP_TOLERANCE * max(primal, 1e-6)) or (not project and info["warnflag"] == 0):
            break
        if not project or attempt == 5:
            raise ValueError(f"SVM optimization needs more precision: {info['task']}; "
                             f"relative duality gap = {gap / max(primal, 1e-6):.3g}")
        # An objective-based stop can be premature on ill-conditioned kernels.
        # Restart at the current solution with a stricter stopping rule.
        start = alpha
        factr = max(1, factr / 100)
        tolerance /= 10
    gradient = objective(alpha)[1]
    projected = gradient.copy()
    projected[(alpha == 0) & (gradient > 0)] = 0
    projected[(alpha == penalty) & (gradient < 0)] = 0
    return {"alpha": alpha, "dual": float(-value), "iterations": info["nit"],
            "projected_gradient": float(np.max(np.abs(projected)))}


def add_primal_cost(result, margins, squared_norm, penalty):
    result["primal"] = float(0.5 * squared_norm + penalty * np.maximum(0, 1 - margins).sum())
    result["gap"] = result["primal"] - result["dual"]
    if result["gap"] > 1e-3 * max(result["primal"], 1e-6):
        raise ValueError("The SVM duality gap is too large; the solution needs more precision.")
    result["support_count"] = int(np.sum(result["alpha"] > 0))
    return result


# 2. Linear SVM: append K and regularize the extended weight vector.

def train_linear(data, labels, penalty, bias_scale=1, initial=None, tolerance=1e-5, factr=np.nan):
    check_training(data, labels, penalty)
    if not np.isfinite(bias_scale) or bias_scale < 0:
        raise ValueError("The bias scale K must be finite and nonnegative.")
    extended = np.vstack([data, np.full(data.shape[1], bias_scale)])
    signs = 2 * labels - 1
    signed_data = extended * signs[None, :]

    def objective(alpha):
        # H = signed_data.T @ signed_data, evaluated without storing an N-by-N matrix.
        weights = signed_data @ alpha
        product = signed_data.T @ weights
        return float(0.5 * (weights @ weights) - alpha.sum()), product - 1

    result = solve_dual(objective, labels.size, penalty, initial, tolerance, factr)
    extended_w = signed_data @ result["alpha"]
    result.update({"w": extended_w[:-1], "b": float(extended_w[-1] * bias_scale),
                   "extended_w": extended_w, "K": bias_scale, "C": penalty})
    margins = signs * (extended_w @ extended)
    return add_primal_cost(result, margins, extended_w @ extended_w, penalty)


def linear_scores(model, data):
    return model["w"] @ data + model["b"]


# 3. Kernels: the constant xi is added after the polynomial or RBF calculation.

def polynomial_kernel(first, second, degree=2, offset=1, bias=0):
    if (not isinstance(degree, (int, np.integer)) or degree < 1 or
            not np.isfinite([offset, bias]).all() or min(offset, bias) < 0):
        raise ValueError("Use a positive integer degree and nonnegative finite c and xi.")
    return (first.T @ second + offset) ** degree + bias


def rbf_kernel(first, second, gamma=1, bias=0):
    if not np.isfinite([gamma, bias]).all() or gamma <= 0 or bias < 0:
        raise ValueError("Use positive finite gamma and nonnegative finite xi.")
    distances = np.sum(first * first, axis=0)[:, None] + np.sum(second * second, axis=0)[None, :]
    distances -= 2 * first.T @ second
    # Roundoff can make the distance of identical samples slightly negative.
    np.maximum(distances, 0, out=distances)
    return np.exp(-gamma * distances) + bias


def train_kernel(data, labels, penalty, kernel, hessian=None, initial=None, tolerance=1e-5, factr=np.nan):
    check_training(data, labels, penalty)
    signs = 2 * labels - 1
    if hessian is None:
        gram = kernel(data, data)
        hessian = gram * signs[:, None] * signs[None, :]
    objective = make_dual_objective(hessian)
    result = solve_dual(objective, labels.size, penalty, initial, tolerance, factr)
    product = hessian @ result["alpha"]
    result.update({"coefficients": result["alpha"] * signs, "training_data": data,
                   "kernel": kernel, "C": penalty})
    return add_primal_cost(result, product, result["alpha"] @ product, penalty)


def kernel_scores(model, data, batch_size=512):
    keep = model["alpha"] > 0
    support_vectors = model["training_data"][:, keep]
    coefficients = model["coefficients"][keep]
    scores = np.empty(data.shape[1])
    for start in range(0, data.shape[1], batch_size):
        end = start + batch_size
        scores[start:end] = coefficients @ model["kernel"](support_vectors, data[:, start:end])
    return scores


def score_metrics(scores, labels, application_prior):
    # Raw SVM scores are not LLRs. Actual DCF here treats them as LLRs only for diagnosis.
    result = evaluate_scores(scores, labels, [(application_prior, 1, 1)])[0]
    result["error_rate"] = float(np.mean((scores > 0) != labels))
    result["scores"] = scores
    return result


# 4. Exercises and reference checks on binary Iris.

def print_result(name, model, metrics):
    print(f"{name:<29} primal {model['primal']:12.6f}; dual {model['dual']:12.6f}; "
          f"gap {model['gap']:.3g}; error {100 * metrics['error_rate']:5.2f}%; "
          f"minDCF {metrics['minimum']:.4f}; actDCF {metrics['actual']:.4f}", flush=True)


def run_iris(data, labels, seed, check_references=True):
    keep = labels != 0
    data = data[:, keep]
    labels = (labels[keep] == 1).astype(np.int32)
    training, validation = split_data(data, labels, seed)
    print("Binary Iris: Virginica = 0, Versicolor = 1", flush=True)
    print(f"Split: {training[1].size} training, {validation[1].size} validation", flush=True)
    results = []
    for bias_scale in [1, 10]:
        for penalty in [0.1, 1.0, 10.0]:
            model = train_linear(*training, penalty, bias_scale)
            scores = linear_scores(model, validation[0])
            metrics = score_metrics(scores, validation[1], 0.5)
            name = f"Linear K={bias_scale}, C={penalty:g}"
            print_result(name, model, metrics)
            results.append({"name": name, "model": model, **metrics})
    configurations = []
    for offset in [0, 1]:
        for bias in [0, 1]:
            kernel = lambda first, second, c=offset, xi=bias: polynomial_kernel(first, second, 2, c, xi)
            configurations.append((f"Poly d=2, c={offset}, xi={bias}", kernel))
    for gamma in [1, 10]:
        for bias in [0, 1]:
            kernel = lambda first, second, g=gamma, xi=bias: rbf_kernel(first, second, g, xi)
            configurations.append((f"RBF gamma={gamma}, xi={bias}", kernel))
    for name, kernel in configurations:
        model = train_kernel(*training, 1.0, kernel)
        metrics = score_metrics(kernel_scores(model, validation[0]), validation[1], 0.5)
        print_result(name, model, metrics)
        results.append({"name": name, "model": model, **metrics})
    if check_references and seed == 0:
        objectives = [3.774974, 15.77993, 78.96893, 2.983576, 11.31707, 54.64483,
                      6.663628, 6.296912, 3.592918, 3.569987, 11.98930, 11.97813, 18.42757, 18.39182]
        minima = [.0556, .0556, .0556, 0, .0556, .0625, .0625, .0625,
                  .0556, .0556, .0625, .0625, .1736, .1736]
        actual = [.0625, .1181, .1181, .2222, .1181, .1181, .1736, .1736,
                  .0556, .0556, .1736, .1736, .2292, .1736]
        errors = [1, 2, 2, 4, 2, 2, 3, 3, 1, 1, 3, 3, 4, 3]
        if not np.allclose([r["model"]["dual"] for r in results], objectives, rtol=0, atol=1e-4):
            raise ValueError("Iris dual objectives differ from the sheet.")
        if (not np.allclose([r["minimum"] for r in results], minima, rtol=0, atol=5e-5) or
                not np.allclose([r["actual"] for r in results], actual, rtol=0, atol=5e-5) or
                not np.allclose([r["error_rate"] for r in results], np.array(errors) / 34, rtol=0, atol=1e-12)):
            raise ValueError("Iris SVM error rates or detection costs differ from the sheet.")
        print("All 14 Iris configurations match the reference objectives and decision metrics.", flush=True)
    else:
        print("Sheet references skipped: they use default Iris data and seed 0.", flush=True)
    return [("iris_svm_comparison", plot_iris(results))]


# 5. Project grids, model cache, and overview plots.

def run_project(data, labels, seed, save_folder=None):
    training, validation = split_data(data, labels, seed)
    mean = training[0].mean(axis=1, keepdims=True)
    signs = 2 * training[1] - 1
    print(f"Project split: {training[1].size} training, {validation[1].size} validation", flush=True)
    experiments = {}
    cache = {"seed": np.array(seed), "linear_cs": LINEAR_CS, "rbf_cs": RBF_CS,
             "rbf_gammas": RBF_GAMMAS, "training_data": training[0],
             "training_labels": training[1], "validation_labels": validation[1],
             "validation_data": validation[0], "training_mean": mean}
    indices = np.random.RandomState(seed).permutation(labels.size)
    cache.update({"training_indices": indices[:training[1].size],
                  "validation_indices": indices[training[1].size:]})

    def save_progress():
        if save_folder is not None:
            save_folder.mkdir(exist_ok=True)
            np.savez_compressed(save_folder / "project_models.npz", **cache)

    # Completed grids can be reused if a long run is interrupted.
    if save_folder is not None and (save_folder / "project_models.npz").is_file():
        with np.load(save_folder / "project_models.npz", allow_pickle=False) as saved:
            if (int(saved["seed"]) == seed and np.array_equal(saved["training_data"], training[0]) and
                    np.array_equal(saved["training_labels"], training[1]) and
                    "validation_data" in saved and np.array_equal(saved["validation_data"], validation[0]) and
                    np.array_equal(saved["validation_labels"], validation[1]) and
                    np.array_equal(saved["linear_cs"], LINEAR_CS) and np.array_equal(saved["rbf_cs"], RBF_CS) and
                    np.array_equal(saved["rbf_gammas"], RBF_GAMMAS)):
                cache.update({key: saved[key] for key in saved.files})

    def restore_grid(key, penalties):
        scores_key = f"{key}_validation_scores"
        if scores_key not in cache or cache[scores_key].shape != (len(penalties), validation[1].size):
            return None
        print(f"Reusing completed {key} grid", flush=True)
        return [{"C": penalty, **score_metrics(scores, validation[1], 0.1)}
                for penalty, scores in zip(penalties, cache[scores_key])]

    for name, key, centered in [("Linear", "linear", False), ("Centered linear", "centered_linear", True)]:
        restored = restore_grid(key, LINEAR_CS)
        if restored is not None:
            experiments[key] = restored
            continue
        train_data = training[0] - mean if centered else training[0]
        val_data = validation[0] - mean if centered else validation[0]
        results, models, initial = [], [], None
        for penalty in LINEAR_CS:
            model = train_linear(train_data, training[1], penalty, initial=initial, tolerance=1e-4, factr=1e5)
            initial = model["alpha"]
            metrics = score_metrics(linear_scores(model, val_data), validation[1], 0.1)
            results.append({"C": penalty, **metrics, "gap": model["gap"]})
            models.append(model)
            print_result(f"{name} C={penalty:g}", model, metrics)
        experiments[key] = results
        cache[f"{key}_w"] = np.vstack([m["w"] for m in models])
        cache[f"{key}_b"] = np.array([m["b"] for m in models])
        cache[f"{key}_alpha"] = np.vstack([m["alpha"] for m in models])
        cache[f"{key}_validation_scores"] = np.vstack([r["scores"] for r in results])
        save_progress()

    kernels = [("Degree-2 polynomial", "poly2", LINEAR_CS,
                lambda first, second: polynomial_kernel(first, second, degree=2))]
    for index, gamma in enumerate(RBF_GAMMAS):
        kernels.append((f"RBF gamma={gamma:.5f}", f"rbf{index}", RBF_CS,
                        lambda first, second, g=gamma: rbf_kernel(first, second, g, bias=1)))
    kernels.append(("Degree-4 polynomial", "poly4", LINEAR_CS,
                    lambda first, second: polynomial_kernel(first, second, degree=4)))
    for name, key, penalties, kernel in kernels:
        restored = restore_grid(key, penalties)
        if restored is not None:
            experiments[key] = restored
            continue
        print(f"\n{name}: computing training and validation kernels", flush=True)
        hessian = kernel(training[0], training[0])
        hessian *= signs[:, None]
        hessian *= signs[None, :]
        validation_kernel = kernel(training[0], validation[0])
        results, coefficients, alphas, initial = [], [], [], None
        for penalty in penalties:
            model = train_kernel(training[0], training[1], penalty, kernel,
                                 hessian=hessian, initial=initial, tolerance=1e-4, factr=1e5)
            initial = model["alpha"]
            scores = model["coefficients"] @ validation_kernel
            metrics = score_metrics(scores, validation[1], 0.1)
            results.append({"C": penalty, **metrics, "gap": model["gap"]})
            coefficients.append(model["coefficients"])
            alphas.append(model["alpha"])
            print_result(f"{key} C={penalty:g}", model, metrics)
        experiments[key] = results
        cache[f"{key}_coefficients"] = np.vstack(coefficients)
        cache[f"{key}_alpha"] = np.vstack(alphas)
        cache[f"{key}_validation_scores"] = np.vstack([r["scores"] for r in results])
        del hessian, validation_kernel
        save_progress()
    print("\nBest validation minDCF per configuration:", flush=True)
    for name, results in experiments.items():
        best = min(results, key=lambda r: r["minimum"])
        actual_best = min(results, key=lambda r: r["actual"])
        print(f"{name:<17} C={best['C']:9.4g}: minDCF={best['minimum']:.6f}, "
              f"actDCF={best['actual']:.6f}; best actual={actual_best['actual']:.6f} "
              f"at C={actual_best['C']:g}", flush=True)
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True)
        path = save_folder / "project_models.npz"
        np.savez_compressed(path, **cache)
        print("Saved models, split metadata, and validation scores:", path, flush=True)
    print("Raw SVM scores are not LLRs; actual DCF assumes that scale for diagnosis only.", flush=True)
    return [("fingerprints_svm_grids", plot_project(experiments)),
            ("fingerprints_feature_products", plot_feature_products(validation[0], validation[1]))]


def plot_iris(results):
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    positions = np.arange(len(results))
    axes[0].bar(positions - .18, [r["minimum"] for r in results], .36, label="Minimum DCF")
    axes[0].bar(positions + .18, [r["actual"] for r in results], .36, label="Actual DCF at prior 0.5")
    axes[0].set(ylabel="Normalized DCF", title="Discrimination and raw-score decisions")
    axes[0].legend()
    axes[1].bar(positions, [100 * r["error_rate"] for r in results], color=COLORS[2])
    axes[1].set(ylabel="Error rate (%)", title="Classification at score threshold zero")
    for axis in axes:
        axis.set_xticks(positions, [r["name"] for r in results], rotation=30, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Binary Iris SVM: seed-0 validation split")
    return figure


def plot_project(experiments):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for axis, key, title in [(axes[0, 0], "linear", "Linear K=1"),
                              (axes[0, 1], "centered_linear", "Centered linear K=1"),
                              (axes[1, 0], "poly2", "Polynomial d=2, c=1, xi=0"),
                              (axes[1, 1], "poly4", "Polynomial d=4, c=1, xi=0")]:
        results = experiments[key]
        axis.semilogx(LINEAR_CS, [r["minimum"] for r in results], "o--", label="Minimum DCF")
        axis.semilogx(LINEAR_CS, [r["actual"] for r in results], "o-", label="Actual DCF")
        axis.set_title(title)
    colors = COLORS + ["#CC79A7"]
    for index, (gamma, color) in enumerate(zip(RBF_GAMMAS, colors)):
        results = experiments[f"rbf{index}"]
        axes[0, 2].semilogx(RBF_CS, [r["minimum"] for r in results], "o-", color=color,
                            label=f"gamma = exp({index - 4})")
        axes[1, 2].semilogx(RBF_CS, [r["actual"] for r in results], "o-", color=color,
                            label=f"gamma = exp({index - 4})")
    axes[0, 2].set_title("RBF xi=1: minimum DCF")
    axes[1, 2].set_title("RBF xi=1: actual DCF")
    for axis in axes.ravel():
        axis.set(xlabel="Penalty C (larger means weaker regularization)", ylabel="Normalized DCF", ylim=(0, None))
        axis.axhline(1, color="gray", linestyle=":")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("Fingerprint SVM validation: application prior 0.1")
    return figure


def plot_feature_products(data, labels):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    product = data[4] * data[5]
    boundaries = np.histogram_bin_edges(product, bins=40)
    for class_id, name, color in zip([0, 1], ["Fake", "Genuine"], COLORS):
        selected = labels == class_id
        axes[0].scatter(data[4, selected], data[5, selected], s=8, alpha=.35, color=color, label=name)
        axes[1].hist(product[selected], bins=boundaries, density=True, alpha=.4, color=color, label=name)
    axes[0].set(xlabel="Feature 5", ylabel="Feature 6", title="Last two original features")
    axes[1].set(xlabel="Feature 5 * feature 6", ylabel="Density", title="A degree-2 feature product")
    for axis in axes:
        axis.legend()
    figure.suptitle("Why higher-degree boundaries can help")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="optional custom Iris CSV or project file")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    lab_folder = Path(__file__).resolve().parent
    folder = lab_folder / "results"
    filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data" / (
        "traindata.txt" if args.project else "iris.csv")
    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            figures = run_project(data, labels, args.seed, folder if args.save else None)
        else:
            data, labels = load_iris(filename)
            figures = run_iris(data, labels, args.seed, args.filename is None)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.error(str(error))
    if args.save:
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
