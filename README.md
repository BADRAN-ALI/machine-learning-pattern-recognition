# Machine Learning Portfolio

A collection of ten Python labs and a fingerprint spoofing detection project, covering statistical modeling, dimensionality reduction, classification, and model evaluation.

**Technologies:** Python · NumPy · SciPy · Matplotlib

## Fingerprint Spoofing Detection

The project classifies fingerprint samples as **fake or genuine** using six extracted numerical features. It compares generative and discriminative models, investigates how feature distributions affect classification, and evaluates whether score calibration and fusion improve decision quality.

The experimental pipeline includes:

- **Data exploration:** feature distributions, class statistics, and covariance analysis.
- **Dimensionality reduction:** principal component analysis (PCA) and linear discriminant analysis (LDA).
- **Classification:** Gaussian classifiers, linear and quadratic logistic regression, polynomial and RBF support vector machines, and Gaussian mixture models.
- **Calibration and fusion:** prior-weighted logistic regression applied to individual and combined classifier scores.
- **Evaluation:** Bayes decisions, normalized detection costs, and analysis on a separate evaluation dataset.

### Experimental Protocol

| Partition | Samples | Purpose |
| --- | ---: | --- |
| Model training | 4,000 | Estimate classifier parameters and feature transformations |
| Validation | 2,000 | Select model settings and assess calibration with five folds |
| Final evaluation | 6,000 | Assess fixed classifiers and calibration maps |

The training/validation split uses seed 0. The primary application has genuine prior **0.1** and unit mistake costs. Final calibration maps are fitted on all validation scores before evaluation data is opened. Evaluation labels are used for reporting, without retraining or changing the selected system.

### Results

Normalized detection cost (DCF) accounts for accepting fake fingerprints and rejecting genuine ones. **Lower values indicate better performance.**

| Calibrated system | Validation actual DCF | Evaluation actual DCF |
| --- | ---: | ---: |
| Gaussian mixture model | **0.1677** | 0.2423 |
| Quadratic logistic regression | 0.2721 | 0.3896 |
| RBF support vector machine | 0.1894 | 0.2889 |
| Three-system fusion | 0.1708 | **0.2203** |

The **calibrated GMM** was selected using validation actual DCF. Fusion achieved the lowest evaluation cost, demonstrating that the development ranking did not fully carry over to the separate data. The original model choice is retained to preserve the evaluation protocol.

The **[project report](project_report.pdf)** presents the methodology, selected configurations, figures, calibration analysis, final evaluation, and limitations.

<details>
<summary>Parameter grids</summary>

- **Logistic regression:** 13 lambda values from `1e-4` to `1e2`, for full linear, reduced linear, weighted linear, and quadratic models.
- **Linear/polynomial SVM:** 11 C values from `1e-5` to `1`, including polynomial degrees 2 and 4.
- **RBF SVM:** 11 C values from `1e-3` to `1e2`; gamma values `exp(-4)`, `exp(-3)`, `exp(-2)`, and `exp(-1)`.
- **GMM:** independently test 1, 2, 4, 8, and 16 components for each class.

</details>

## Laboratory Modules

All ten labs are complete. Each has one main Python file, with related inputs and figures stored in `data/` and `results/` where applicable.

| Module | Topics |
| --- | --- |
| [Lab 01](lab01/lab01.py) | Python fundamentals and numerical computing; eight exercises |
| [Lab 02](lab02/lab02.py) | Dataset loading, descriptive statistics, and visualization |
| [Lab 03](lab03/lab03.py) | PCA, LDA, and dimensionality reduction experiments |
| [Lab 04](lab04/lab04.py) | Gaussian densities and maximum-likelihood estimation |
| [Lab 05](lab05/lab05.py) | Multivariate Gaussian, tied-covariance, and Naive Bayes classifiers |
| [Lab 06](lab06/lab06.py) | Bayes decisions, detection costs, and ROC analysis |
| [Lab 07](lab07/lab07.py) | Binary, weighted, quadratic, and multiclass logistic regression |
| [Lab 08](lab08/lab08.py) | Linear, polynomial, and RBF support vector machines |
| [Lab 09](lab09/lab09.py) | Gaussian mixtures, EM estimation, and LBG initialization |
| [Lab 10](lab10/lab10.py) | Calibration, score fusion, final evaluation, and model audit |

## Installation

Use **Python 3.10 or newer**. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows Command Prompt or `source .venv/bin/activate` on Linux/macOS, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

After providing the local Iris CSV, run the data exploration example:

```bash
python lab02/lab02.py --save --no-show
```

Run the fingerprint project's final calibration, fusion, and evaluation:

```bash
python lab10/lab10.py --project --save --no-show
```

Labs 2-10 run their standard exercises without `--project`. For fingerprint experiments, add `--project` to Labs 3-10; Lab 2 also requires `lab02/data/traindata.txt` as its positional argument. Lab 1 takes an exercise number, for example `python lab01/lab01.py 7`. Use `--help` for the available options.

In Labs 2-10, `--save` writes figures to the module's `results/` folder, and `--no-show` disables plot windows. Keep the lab folders together: later modules reuse earlier implementations.

### Data and Reproducibility

No datasets, databases, supplied reference arrays, or fitted model caches are distributed with the repository. All `data/` folders are excluded from version control, including the Iris CSV. Data-dependent experiments require the relevant inputs to be obtained separately and stored locally.

For fingerprint experiments, use `lab02/data/traindata.txt` and `lab02/data/evaldata.txt`. For Iris examples, use `lab02/data/iris.csv`. Restore the supplied exercise inputs and reference arrays in each module's local `data/` folder where required.

Lab 10 requires the saved model caches from Labs 7-9. To generate them, run each of those modules with `--project --save --no-show` using the same data and default seed. Kernel SVM training is the most computationally demanding stage.

The source code, saved overview figures, and project report document the experiments. The report also discusses the limitations of selecting model settings on one validation split and the distinction between minimum and achieved detection cost.
