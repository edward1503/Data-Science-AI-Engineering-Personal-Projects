# Market Basket Analysis using ECLAT Algorithm - A Frequent Pattern Approach

<!-- centers image, keeps it responsive with a max width -->
<img src="figures/frequentpatternviz.jpg" alt="Frequent pattern visualization"
     style="display:block; margin-left:auto; margin-right:auto; max-width:90%; height:auto;">

This repository demonstrates market-basket analysis using the ECLAT algorithm to discover frequent itemsets from transaction data. The project contains example datasets, preprocessing utilities, an implementation of the ECLAT algorithm, and Jupyter notebooks for exploratory data analysis and experiments.

**Key goals:**
- Implement ECLAT (vertical-format frequent pattern mining).
- Provide reusable preprocessing for transaction data.
- Demonstrate end-to-end analysis in notebooks with visualizations and examples.

**Contents at a glance:**
- `dataset/` — raw and sparse transaction CSVs used for experiments.
- `notebooks/` — exploratory analysis and ECLAT example notebooks.
- `scripts/` — Python scripts for preprocessing and ECLAT implementation.

**Recommended audience:** data scientists and students learning association rule mining and frequent pattern mining.

**Quick Links:**
- Notebook: `notebooks/ECLAT.ipynb`
- Preprocessing EDA notebook: `notebooks/Preprocessing_EDA.ipynb`
- ECLAT script: `scripts/eclat.py`
- Preprocessing script: `scripts/process.py`

**License & Citation**: Add a license file if you plan to publish or share this repository publicly.

---

**Repository Structure**

- `dataset/`
	- `list/` — transaction-style CSVs (one row per transaction; items separated by commas or other delimiters). Examples: `Groceries 2.csv`, `Market Basket Analysis 1.csv`.
	- `sparse/` — datasets in sparse/TF format (one item per row or one-hot style CSVs). Examples: `Books.csv`, `Groceries 3.csv`.

- `notebooks/`
	- `ECLAT.ipynb` — step-by-step ECLAT implementation, examples, and visualizations.
	- `Preprocessing_EDA.ipynb` — data cleaning, transformation from basket-list to transaction format, and exploratory plots.

- `scripts/`
	- `eclat.py` — script implementing the ECLAT algorithm; can be used as a module or run from the command line.
	- `process.py` — preprocessing utilities for reading different CSV formats and converting to the format ECLAT expects.

**Setup / Requirements**

- Python 3.8+ recommended.
- Typical packages used in notebooks and scripts: `pandas`, `numpy`, `matplotlib`, `seaborn`. If you run the notebooks, install Jupyter as well.

Create a virtual environment named `uv` and install dependencies (Windows `cmd.exe`):

```
python -m venv uv
uv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to run code from `scripts/`, ensure the working directory is the repository root.

**How to run the notebooks**

- Start Jupyter from the repository root (Windows `cmd.exe`):

```
uv\Scripts\activate
jupyter notebook
```

- Open `notebooks/ECLAT.ipynb` to step through preprocessing, model execution, and visualization.

**Running scripts**

- Example: running `eclat.py` directly (if it supports CLI). From the repo root in `cmd.exe`:

```
uv\Scripts\activate
python scripts\eclat.py --input "dataset\list\Market Basket Analysis 1.csv" --minsup 0.02
```

Adjust flags according to the script's implemented CLI (see `scripts/eclat.py` source for supported arguments).

**Preprocessing notes**

- Use `scripts/process.py` to convert datasets into transaction lists (one row per transaction, items comma-separated). Typical steps:
	- Normalize item names (lowercase, trim whitespace).
	- Remove duplicates per transaction if required.
	- Export cleaned CSV for notebook experiments.

**Examples & Tips**

- If you have very large datasets, prefer the sparse/vertical representation to reduce memory usage.
- When tuning `minsup`, start with a higher value (e.g., 0.02) and lower it progressively to discover more patterns.

**Contributing**

- Suggestions and improvements are welcome. Consider adding:
	- Tests for `scripts/eclat.py`.
	- A `requirements.txt` or `environment.yml` for reproducible environments.
	- A CLI help message if not present.

**Next steps (suggested)**

- Add a `requirements.txt` with pinned versions.
- Add a small example `run.bat` demonstrating a full pipeline run (preprocess -> eclat -> results).

If you'd like, I can:
- generate a `requirements.txt` from the notebooks and scripts,
- run quick static checks on `scripts/eclat.py` and `scripts/process.py`, or
- add a small `run.bat` example for Windows.

---

If you want any of the suggested next steps, tell me which and I'll implement it.
