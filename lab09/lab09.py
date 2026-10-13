"""Lab 9: Gaussian mixtures, EM, LBG, and class-conditional density models."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

LABS_FOLDER = Path(__file__).resolve().parents[1]
for folder in ["lab02", "lab03", "lab04", "lab06"]:
    sys.path.insert(0, str(LABS_FOLDER / folder))
from lab02 import load_iris, load_fingerprints, COLORS
from lab03 import split_data
from lab04 import fit_gaussian, logpdf_GAU_ND
from lab06 import evaluate_scores, bayes_error_curve

COMPONENT_COUNTS = [1, 2, 4, 8, 16]
COVARIANCE_TYPES = ["full", "diagonal", "tied"]


# 1. A mixture is a list of (weight, column mean, covariance) tuples.

def check_gmm(gmm, dimensions):
    if not gmm:
        raise ValueError("A GMM needs at least one component.")
    weights = np.array([component[0] for component in gmm])
    if (not np.isfinite(weights).all() or np.any(weights <= 0) or
            not np.isclose(weights.sum(), 1, rtol=1e-10, atol=1e-12)):
        raise ValueError("Mixture weights must be positive and sum to one.")
    for weight, mean, covariance in gmm:
        if mean.shape != (dimensions, 1) or covariance.shape != (dimensions, dimensions):
            raise ValueError("Component parameter shapes must match the feature dimension.")


def load_gmm(filename):
    components = json.loads(Path(filename).read_text())
    gmm = [(float(weight), np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float))
           for weight, mean, covariance in components]
    if not gmm:
        raise ValueError("The GMM file is empty.")
    check_gmm(gmm, gmm[0][1].shape[0])
    return gmm


def component_log_scores(data, gmm):
    check_gmm(gmm, data.shape[0])
    return np.vstack([logpdf_GAU_ND(data, mean, covariance) + np.log(weight)
                      for weight, mean, covariance in gmm])


def logpdf_GMM(data, gmm):
    """Return one log density per sample column, with shape (N,)."""
    return logsumexp(component_log_scores(data, gmm), axis=0)


def responsibilities(data, gmm):
    scores = component_log_scores(data, gmm)
    log_density = logsumexp(scores, axis=0)
    return np.exp(scores - log_density[None, :]), log_density


# 2. Covariance restrictions are applied before eigenvalue flooring.

def floor_covariance(covariance, psi):
    covariance = (covariance + covariance.T) / 2
    if psi is None:
        return covariance
    if not np.isfinite(psi) or psi <= 0:
        raise ValueError("The covariance floor psi must be finite and positive, or None.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors * np.maximum(eigenvalues, psi)[None, :]) @ eigenvectors.T


def constrain_gmm(gmm, covariance_type="full", psi=None):
    if covariance_type not in COVARIANCE_TYPES:
        raise ValueError(f"Unknown covariance type: {covariance_type}")
    updated = [(weight, mean.copy(), covariance.copy()) for weight, mean, covariance in gmm]
    if covariance_type == "diagonal":
        updated = [(weight, mean, np.diag(np.diag(covariance)))
                   for weight, mean, covariance in updated]
    elif covariance_type == "tied":
        shared = sum(weight * covariance for weight, mean, covariance in updated)
        updated = [(weight, mean, shared.copy()) for weight, mean, covariance in updated]
    return [(weight, mean, floor_covariance(covariance, psi)) for weight, mean, covariance in updated]


# 3. EM: soft component assignments, then weighted Gaussian estimates.

def em_step(data, gmm, covariance_type="full", psi=None):
    posterior, _ = responsibilities(data, gmm)
    updated = []
    for row in posterior:
        count = row.sum()
        if not np.isfinite(count) or count <= 0:
            raise ValueError("A mixture component has no effective samples.")
        mean = (data @ row[:, None]) / count
        centered = data - mean
        covariance = (centered * row[None, :]) @ centered.T / count
        updated.append((float(count / data.shape[1]), mean, covariance))
    return constrain_gmm(updated, covariance_type, psi)


def train_em(data, initial_gmm, covariance_type="full", psi=None, threshold=1e-6, max_iterations=20000):
    if (data.ndim != 2 or min(data.shape) == 0 or not np.isfinite(data).all() or
            not np.isfinite(threshold) or threshold <= 0 or max_iterations < 1):
        raise ValueError("EM needs finite sample columns, a positive threshold, and an iteration limit.")
    check_gmm(initial_gmm, data.shape[0])
    gmm = constrain_gmm(initial_gmm, covariance_type, psi)
    history = [float(logpdf_GMM(data, gmm).mean())]
    for iteration in range(max_iterations):
        updated = em_step(data, gmm, covariance_type, psi)
        likelihood = float(logpdf_GMM(data, updated).mean())
        change = likelihood - history[-1]
        if change < -1e-9:
            raise ValueError(f"EM average log likelihood decreased by {change:.3g}.")
        history.append(likelihood)
        gmm = updated
        if change <= threshold:
            return gmm, np.array(history)
    raise ValueError("EM did not converge within the iteration limit.")


# 4. LBG: split each component along its largest-variance direction, then fit EM.

def split_gmm(gmm, alpha=0.1):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("The LBG displacement alpha must be finite and positive.")
    result = []
    for weight, mean, covariance in gmm:
        eigenvectors, eigenvalues, _ = np.linalg.svd(covariance)
        displacement = eigenvectors[:, :1] * np.sqrt(eigenvalues[0]) * alpha
        result.extend([(weight / 2, mean - displacement, covariance.copy()),
                       (weight / 2, mean + displacement, covariance.copy())])
    return result


def train_lbg(data, components, covariance_type="full", psi=0.01, alpha=0.1, threshold=1e-6):
    if not isinstance(components, (int, np.integer)) or components < 1 or components & (components - 1):
        raise ValueError("LBG requires a positive power-of-two component count.")
    mean, covariance = fit_gaussian(data)
    gmm = constrain_gmm([(1.0, mean, covariance)], covariance_type, psi)
    histories = [np.array([logpdf_GMM(data, gmm).mean()])]
    while len(gmm) < components:
        gmm, history = train_em(data, split_gmm(gmm, alpha), covariance_type, psi, threshold)
        histories.append(history)
    return gmm, histories


def train_class_models(data, labels, components, covariance_type="full", psi=0.01):
    if labels.shape != (data.shape[1],) or np.unique(labels).size < 2:
        raise ValueError("Classification needs one label per sample and at least two classes.")
    return {int(class_id): train_lbg(data[:, labels == class_id], components, covariance_type, psi)[0]
            for class_id in np.unique(labels)}


def classifier_scores(data, models):
    classes = np.array(sorted(models), dtype=np.int32)
    scores = np.vstack([logpdf_GMM(data, models[int(class_id)]) for class_id in classes])
    return scores, classes


def binary_scores(data, models):
    if set(models) != {0, 1}:
        raise ValueError("Binary GMM scores require class models 0 and 1.")
    return logpdf_GMM(data, models[1]) - logpdf_GMM(data, models[0])


# 5. Density references, Iris classification, and the supplied binary dataset.

def run_density_references(folder):
    results = []
    for dimension in [1, 4]:
        name = f"{dimension}D"
        data = np.load(folder / f"GMM_data_{name}.npy", allow_pickle=False)
        initial = load_gmm(folder / f"GMM_{name}_3G_init.json")
        expected = np.load(folder / f"GMM_{name}_3G_init_ll.npy", allow_pickle=False).ravel()
        calculated = logpdf_GMM(data, initial)
        difference = np.max(np.abs(calculated - expected))
        print(f"{name} initial log-density maximum difference: {difference:.3e}", flush=True)
        if not np.allclose(calculated, expected, rtol=0, atol=1e-10):
            raise ValueError(f"{name} density reference check failed.")
        em, history = train_em(data, initial)
        lbg, lbg_histories = train_lbg(data, 4, psi=None)
        for description, fitted, reference in [
                ("3-component EM", em, load_gmm(folder / f"GMM_{name}_3G_EM.json")),
                ("4-component LBG", lbg, load_gmm(folder / f"GMM_{name}_4G_EM_LBG.json"))]:
            likelihood = float(logpdf_GMM(data, fitted).mean())
            reference_likelihood = float(logpdf_GMM(data, reference).mean())
            print(f"{name} {description}: average LL = {likelihood:.8f}; "
                  f"reference = {reference_likelihood:.8f}", flush=True)
            if abs(likelihood - reference_likelihood) > 2e-6:
                raise ValueError(f"{name} {description} differs from the reference likelihood.")
        results.append({"data": data, "initial": initial, "em": em, "history": history,
                        "lbg": lbg, "lbg_histories": lbg_histories})
    return results


def classification_grid(data, labels, seed, binary=False):
    training, validation = split_data(data, labels, seed)
    results = {}
    for covariance_type in COVARIANCE_TYPES:
        # Continue each class's LBG sequence rather than retraining its smaller models.
        class_models = {}
        rows = []
        for components in COMPONENT_COUNTS:
            for class_id in np.unique(training[1]):
                samples = training[0][:, training[1] == class_id]
                if components == 1:
                    class_models[int(class_id)] = train_lbg(samples, 1, covariance_type)[0]
                else:
                    class_models[int(class_id)] = train_em(
                        samples, split_gmm(class_models[int(class_id)]), covariance_type, psi=0.01)[0]
            scores, classes = classifier_scores(validation[0], class_models)
            predictions = classes[np.argmax(scores, axis=0)]
            row = {"components": components, "errors": int(np.sum(predictions != validation[1])),
                   "error_rate": float(np.mean(predictions != validation[1]))}
            if binary:
                row.update(evaluate_scores(scores[1] - scores[0], validation[1], [(.5, 1, 1)])[0])
            rows.append(row)
            print(f"{covariance_type:<10} components {components:>2}: errors "
                  f"{row['errors']}/{validation[1].size}, rate {100 * row['error_rate']:.2f}%" +
                  (f", minDCF {row['minimum']:.4f}, actDCF {row['actual']:.4f}" if binary else ""), flush=True)
        results[covariance_type] = rows
    return results


def run_lab(iris_data, iris_labels, seed, folder, check_iris=True):
    densities = run_density_references(folder)
    print("\nIris classification:", flush=True)
    iris = classification_grid(iris_data, iris_labels, seed)
    external_data = np.load(folder / "ext_data_binary.npy", allow_pickle=False)
    external_labels = np.load(folder / "ext_data_binary_labels.npy", allow_pickle=False)
    print("\nAdditional binary dataset (application prior 0.5):", flush=True)
    binary = classification_grid(external_data, external_labels, seed, binary=True)
    if seed == 0:
        if check_iris:
            expected = [[2, 2, 2, 2, 2], [2, 2, 3, 1, 2], [2, 2, 2, 2, 3]]
            for covariance_type, errors in zip(COVARIANCE_TYPES, expected):
                if [row["errors"] for row in iris[covariance_type]] != errors:
                    raise ValueError(f"Iris {covariance_type} differs from the reference errors.")
        expected_min = [[.4984, .4302, .5195, .5804, .6364],
                        [.5203, .4643, .4213, .4781, .4870], [.4984, .4984, .4416, .4278, .4383]]
        expected_actual = [[.5398, .4416, .5706, .6177, .6640],
                           [.5625, .4643, .4513, .4781, .5446], [.5398, .5398, .4643, .4846, .5252]]
        for covariance_type, minima, actual in zip(COVARIANCE_TYPES, expected_min, expected_actual):
            if (not np.allclose([r["minimum"] for r in binary[covariance_type]], minima, atol=5e-5, rtol=0) or
                    not np.allclose([r["actual"] for r in binary[covariance_type]], actual, atol=5e-5, rtol=0)):
                raise ValueError(f"Binary {covariance_type} differs from the reference costs.")
        print("\nAll applicable classification reference checks passed.", flush=True)
    return [("gmm_density_estimation", plot_densities(densities)),
            ("gmm_classification", plot_classification(iris, binary))]


# 6. Project: choose the component count independently for fake and genuine.

def pack_gmm(cache, prefix, gmm):
    cache[f"{prefix}_weights"] = np.array([weight for weight, mean, covariance in gmm])
    cache[f"{prefix}_means"] = np.stack([mean for weight, mean, covariance in gmm])
    cache[f"{prefix}_covariances"] = np.stack([covariance for weight, mean, covariance in gmm])


def unpack_gmm(cache, prefix):
    return [(float(weight), mean, covariance) for weight, mean, covariance in zip(
        cache[f"{prefix}_weights"], cache[f"{prefix}_means"], cache[f"{prefix}_covariances"])]


def previous_candidates(training, validation, validation_indices, seed):
    candidates = []
    for lab, groups in [("lab07", ["linear", "weighted", "quadratic"]),
                         ("lab08", ["linear", "centered_linear", "poly2", "poly4", "rbf0", "rbf1", "rbf2", "rbf3"])]:
        path = LABS_FOLDER / lab / "results/project_models.npz"
        if not path.is_file():
            raise ValueError(f"Run python {lab}/{lab}.py --project --save first to build the comparison cache.")
        with np.load(path, allow_pickle=False) as saved:
            if (int(saved["seed"]) != seed or not np.array_equal(saved["validation_indices"], validation_indices) or
                    not np.array_equal(saved["validation_labels"], validation[1])):
                raise ValueError(f"The {lab} cache uses a different split. Rebuild it with --seed {seed}.")
            if ("training_data" not in saved or "validation_data" not in saved or
                    not np.array_equal(saved["training_data"], training[0]) or
                    not np.array_equal(saved["training_labels"], training[1]) or
                    not np.array_equal(saved["validation_data"], validation[0])):
                raise ValueError(f"The {lab} cache needs the current project data. Rebuild it with --project --save.")
            best = None
            for group in groups:
                suffix = "validation_llr" if lab == "lab07" else "validation_scores"
                for index, scores in enumerate(saved[f"{group}_{suffix}"]):
                    result = evaluate_scores(scores, validation[1], [(.1, 1, 1)])[0]
                    if best is None or result["minimum"] < best["minimum"]:
                        parameter = saved["lambdas"][index] if lab == "lab07" else (
                            saved["rbf_cs"][index] if group.startswith("rbf") else saved["linear_cs"][index])
                        best = {"name": "Logistic regression" if lab == "lab07" else "SVM",
                                "description": f"{group}, {'lambda' if lab == 'lab07' else 'C'}={parameter:g}",
                                "scores": scores.copy(), **result}
            candidates.append(best)
    return candidates


def run_project(data, labels, seed, save_folder=None):
    training, validation = split_data(data, labels, seed)
    indices = np.random.RandomState(seed).permutation(labels.size)
    validation_indices = indices[training[1].size:]
    cache = {"seed": np.array(seed), "component_counts": np.array(COMPONENT_COUNTS),
             "training_indices": indices[:training[1].size], "validation_indices": validation_indices,
             "training_data": training[0], "training_labels": training[1],
             "validation_data": validation[0], "validation_labels": validation[1], "psi": np.array(.01)}
    if save_folder is not None and (save_folder / "project_models.npz").is_file():
        with np.load(save_folder / "project_models.npz", allow_pickle=False) as saved:
            if (int(saved["seed"]) == seed and np.array_equal(saved["training_data"], training[0]) and
                    np.array_equal(saved["training_labels"], training[1]) and
                    np.array_equal(saved["validation_data"], validation[0]) and
                    np.array_equal(saved["validation_labels"], validation[1]) and
                    np.array_equal(saved["component_counts"], COMPONENT_COUNTS) and float(saved["psi"]) == .01):
                cache.update({key: saved[key] for key in saved.files})
    class_scores = []
    print(f"Project split: {training[1].size} training, {validation[1].size} validation", flush=True)
    for class_id in [0, 1]:
        samples = training[0][:, training[1] == class_id]
        rows, gmm = [], None
        for components in COMPONENT_COUNTS:
            prefix = f"class{class_id}_{components}"
            if f"{prefix}_weights" in cache:
                gmm = unpack_gmm(cache, prefix)
                print(f"Reusing class {class_id}, {components} components", flush=True)
            elif components == 1:
                gmm = train_lbg(samples, 1)[0]
            else:
                gmm, history = train_em(samples, split_gmm(gmm), psi=.01)
                print(f"Class {class_id}, {components} components: {len(history)-1} EM updates; "
                      f"average training LL {history[-1]:.6f}", flush=True)
            pack_gmm(cache, prefix, gmm)
            rows.append(logpdf_GMM(validation[0], gmm))
        class_scores.append(np.vstack(rows))
    minimum = np.empty((5, 5))
    actual = np.empty((5, 5))
    all_scores = np.empty((5, 5, validation[1].size))
    best = None
    print("\nFake components / genuine components: minDCF / actDCF at prior 0.1", flush=True)
    for first, fake_count in enumerate(COMPONENT_COUNTS):
        for second, genuine_count in enumerate(COMPONENT_COUNTS):
            scores = class_scores[1][second] - class_scores[0][first]
            all_scores[first, second] = scores
            result = evaluate_scores(scores, validation[1], [(.1, 1, 1)])[0]
            minimum[first, second] = result["minimum"]
            actual[first, second] = result["actual"]
            print(f"{fake_count:>2} / {genuine_count:>2}: {result['minimum']:.4f} / {result['actual']:.4f}", flush=True)
            if best is None or result["minimum"] < best["minimum"]:
                best = {"name": "GMM", "description": f"fake={fake_count}, genuine={genuine_count}",
                        "scores": scores.copy(), **result}
    cache.update({"class0_validation_logpdf": class_scores[0], "class1_validation_logpdf": class_scores[1],
                  "validation_llr": all_scores, "minimum_dcf": minimum, "actual_dcf": actual})
    if save_folder is not None:
        save_folder.mkdir(exist_ok=True)
        np.savez_compressed(save_folder / "project_models.npz", **cache)
    comparisons = [best] + previous_candidates(training, validation, validation_indices, seed)
    print("\nSelected model comparison:", flush=True)
    for result in comparisons:
        print(f"{result['name']:<20} {result['description']}: minDCF={result['minimum']:.6f}; "
              f"actDCF={result['actual']:.6f}", flush=True)
    log_odds = np.linspace(-4, 4, 41)
    for result in comparisons:
        result["actual_curve"], result["minimum_curve"] = bayes_error_curve(result["scores"], validation[1], log_odds)
        cache[f"selected_{result['name'].split()[0].lower()}_scores"] = result["scores"]
    cache["bayes_log_odds"] = log_odds
    if save_folder is not None:
        np.savez_compressed(save_folder / "project_models.npz", **cache)
        print("Saved all class GMMs, validation scores, and selected comparison scores.", flush=True)
    return [("fingerprints_gmm_selection", plot_project_grid(minimum, actual)),
            ("fingerprints_model_bayes_curves", plot_comparison(comparisons, log_odds))]


# 7. Four compact figures cover density estimation, classification, and the project.

def plot_densities(results):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    result = results[0]
    data = result["data"]
    axis = axes[0, 0]
    axis.hist(data.ravel(), bins=50, density=True, alpha=.3, label="1D samples")
    grid = np.linspace(data.min()-1, data.max()+1, 700)[None,:]
    for name, key in [("Initial 3 components", "initial"), ("EM 3 components", "em"), ("LBG 4 components", "lbg")]:
        axis.plot(grid.ravel(), np.exp(logpdf_GMM(grid, result[key])), label=name)
    axis.set(xlabel="Sample value", ylabel="Density", title="1D fitted densities")
    axis.legend(fontsize=8)
    for axis, result in zip([axes[0, 1], axes[1, 0]], results):
        axis.plot(result["history"])
        axis.set(xlabel="EM update", ylabel="Average log likelihood", title=f"{result['data'].shape[0]}D: 3-component EM")
        axis.grid(alpha=.2)
    for result, color in zip(results, COLORS):
        offset = 0
        for index, history in enumerate(result["lbg_histories"][1:]):
            axes[1, 1].plot(np.arange(len(history))+offset, history, color=color,
                            linestyle="-" if index==0 else "--",
                            label=f"{result['data'].shape[0]}D: {2**(index+1)} components")
            offset += len(history)
    axes[1, 1].set(xlabel="Updates within successive LBG stages", ylabel="Average log likelihood", title="LBG: separate EM stages")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("GMM density estimation and EM convergence")
    return figure


def plot_classification(iris, binary):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for covariance_type, color in zip(COVARIANCE_TYPES, COLORS):
        axes[0].plot(COMPONENT_COUNTS, [100*r["error_rate"] for r in iris[covariance_type]], "o-", color=color, label=covariance_type)
        axes[1].plot(COMPONENT_COUNTS, [r["minimum"] for r in binary[covariance_type]], "o-", color=color, label=covariance_type)
        axes[2].plot(COMPONENT_COUNTS, [r["actual"] for r in binary[covariance_type]], "o-", color=color, label=covariance_type)
    for axis, title, ylabel in zip(axes, ["Iris validation error", "Binary data: minimum DCF", "Binary data: actual DCF"],
                                   ["Error rate (%)", "Normalized DCF", "Normalized DCF"]):
        axis.set(xlabel="Components per class", title=title, ylabel=ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(COMPONENT_COUNTS, COMPONENT_COUNTS)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    return figure


def plot_project_grid(minimum, actual):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, values, title in zip(axes, [minimum, actual], ["Minimum DCF", "Actual DCF"]):
        image = axis.imshow(values, cmap="viridis_r", vmin=0, vmax=max(minimum.max(),actual.max()))
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{values[row,column]:.3f}", ha="center", va="center",
                          color="white" if values[row,column]>max(minimum.max(),actual.max())/2 else "black")
        axis.set_xticks(range(5), COMPONENT_COUNTS)
        axis.set_yticks(range(5), COMPONENT_COUNTS)
        axis.set(xlabel="Genuine components", ylabel="Fake components", title=title)
        figure.colorbar(image, ax=axis, shrink=.8)
    figure.suptitle("Fingerprint full-covariance GMMs: application prior 0.1")
    return figure


def plot_comparison(comparisons, log_odds):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for result, color in zip(comparisons, COLORS):
        axes[0].plot(log_odds, result["minimum_curve"], color=color, label=result["name"])
        axes[1].plot(log_odds, result["actual_curve"], color=color, label=result["name"])
    for axis, title in zip(axes, ["Discrimination: minimum DCF", "Decisions: actual DCF"]):
        axis.set(xlabel="Effective prior log odds", title=title)
        axis.axhline(1, color="gray", linestyle=":")
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Normalized DCF")
    figure.suptitle("Selected GMM, logistic regression, and SVM models")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="optional custom Iris CSV or project file")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    results_folder = folder / "results"
    filename = Path(args.filename) if args.filename else LABS_FOLDER / "lab02/data" / (
        "traindata.txt" if args.project else "iris.csv")
    try:
        if args.project:
            data, labels = load_fingerprints(filename)
            figures = run_project(data, labels, args.seed, results_folder if args.save else None)
        else:
            data, labels = load_iris(filename)
            figures = run_lab(data, labels, args.seed, folder / "data", args.filename is None)
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
