# Data requirements

Supply your authorized original course files locally in this directory. These files are not bundled or downloaded automatically.

| File | Expected fields and units |
|---|---|
| `insurance_us.csv` | Comma-delimited: Gender, Age, Salary, Family members, Insurance benefits. No missing, nonnumeric or negative values in modeled columns. Records are not printed by the revised pipeline. |

The notebook's synthetic example runs without these files. To analyze the original dataset, set `RUN_ORIGINAL_DATA = True` in `notebooks/analysis.ipynb` after adding them. The corrected implementation will compute new results; previous empirical outputs are not reused.

CSV, TSV, spreadsheets and this directory's datasets are excluded from Git by default. No permission to redistribute the original datasets is implied.
