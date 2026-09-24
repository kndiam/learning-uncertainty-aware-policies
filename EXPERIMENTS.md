# Running the thesis experiments

All commands run from the repository root. Tests: `python -m pytest -q tests/`.

## Settings
`ablation_utils/config.py` holds everything that must agree across runs:
datasets and their EU thinning, base models, lambda per dataset, and the
coverage weight `c(u) = 1 + GAMMA * u` (constant control: `c = 1`).

## EU adaptivity, several seeds
```
python -m scripts.run_seeds --task clf --dataset PENDIGITS --seeds 0 1 2 3 4
python -m scripts.run_seeds --task reg --dataset CALIFORNIA --seeds 0 1 2 3 4
python -m scripts.summarize
```
Add `--learned` to also train the policy nets (slower). Results go to
`results/*.npz` (git-ignored); tables to `results/tables/`.

Policies stored per method: `exact`, `exact_bounded` (exact on the net's
clipped domain), `smoothed` (optimum of the sigmoid surrogate the net trains
on), and `learned` with `--learned`.

## Learned vs exact
- Toy decomposition: `classification/classification_closed_form_evaluation.ipynb`,
  section "Where does the learned-vs-exact gap come from?".
- On real data: `run_seeds --learned`, then the `_gap.csv` table from `summarize`.
- Cost vs test-set size: `python -m scripts.time_policies --dataset PENDIGITS`.
