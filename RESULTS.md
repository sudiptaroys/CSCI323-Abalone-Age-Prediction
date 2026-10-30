# Results, provenance and verification

## Computed verification run

The full `python run_all.py` path completed, including the ten-candidate five-fold RF search, all model fits, subgroup fits and aggregate consistency checks. Environment: Python 3.12.14, Windows; package versions and complete fitted model parameters are recorded in `verification_environment.json`. Warnings are retained in that file (0 recorded).

| Method | MAE (years) | MSE (years²) | RMSE (years) | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.593107 | 4.891232 | 2.211613 | 0.548163 |
| Random Forest (default) | 1.582703 | 5.107362 | 2.259947 | 0.528197 |
| Random Forest (tuned) | 1.547888 | 4.796975 | 2.190200 | 0.556870 |
| XGBoost | 1.557844 | 4.950650 | 2.225006 | 0.542674 |
| MLP | 1.499916 | 4.434160 | 2.105744 | 0.590386 |


**MLP has the lowest measured MAE/MSE and highest R² in this run.** Its rounded scores match the original report, as do the default and tuned Random Forest scores. Linear regression supplies an additional baseline whose aggregate metrics were not printed in the original notebook. XGBoost's current score differs slightly from the original report snapshot and the public prior notebook's saved output.

## Three separate XGBoost records

| Source | MAE | MSE | R² |
|---|---:|---:|---:|
| Original report screenshot | 1.5599 | 4.9603 | 0.5418 |
| Prior public notebook saved output | 1.561547 | 4.987155 | 0.539302 |
| Current verification run | 1.557844 | 4.950650 | 0.542674 |

These are distinct records, not interchangeable values. The original report screenshot is the source for `report_metrics.csv`; it was not regenerated from the available public notebook. Library versions and implementation changes can affect boosted-tree runs, but the exact cause of the historical discrepancy is **unverified** because the original software environment is unavailable. The fresh results are stored separately and do not overwrite the report record. No result was selected or changed to make MLP win.

## Checks performed

- Official dataset shape: 4,177 × 9; eight original predictors, 10 encoded inputs, no missing values or duplicate records. The audit records the original file's SHA-256.
- Train/test partitions are disjoint; the exact 3,341/836 row IDs are in `split_manifest.csv`. Numeric standardization uses training statistics only. Test sex counts: F=262, I=267, M=307.
- The target is Rings + 1.5 exactly once. Rings never appears among the input features.
- RF parameter selection uses only five-fold training CV; the same held-out rows evaluate every method. The search results and selected parameters are saved.
- For RF, XGBoost and MLP, sample-weighted subgroup MAE/MSE reproduce the corresponding overall test metrics. R² is recomputed globally; subgroup R² values must not be averaged as if they were a global score.
- Full scripts executed successfully; notebook code cells were checked for valid Python. The notebooks are provided with cleared execution outputs so opening them does not misrepresent a cached result as a live run.

## Interpretation and limits

The MLP captures nonlinear relationships and produces the strongest result on this one split. Its MAE of approximately 1.50 years still reflects meaningful individual prediction error; R² around 0.59 leaves substantial unexplained variation. Larger residuals among older animals and differences across F/I/M are useful diagnostics, not proof that membership causes error.

The tuned Random Forest improves on its default configuration. The linear baseline's MSE is lower than that of the default forest and the current XGBoost run, illustrating that a more complex model need not improve every metric. MAE and MSE weight errors differently, so rankings can differ by metric.

Historical RF global/local analysis compares a tuned global forest against default local forests; parameter and population effects are confounded. XGBoost/MLP local fits use the same estimator settings as their global counterpart, but smaller training sizes still affect variance. The verification RF age-bin definitions cover ages above 25 and show empty intervals explicitly. This split has no test cases above 25, so the extra interval does not change the measured errors; the oldest dataset observations are in training.

Further evidence would require repeated or nested evaluation, uncertainty intervals and a controlled subgroup study. Those are proposed improvements, not experiments claimed to have been completed. The related research paper's three-class accuracy cannot be compared directly with this regression task.
