# Data

Raw data is **not** committed (size, Kaggle licence). `data/` is git-ignored except this file.

## Automatic

```bash
kaggle auth login          # once. Browser OAuth flow; nothing to paste
# also accept the rules once: https://www.kaggle.com/competitions/home-credit-default-risk (Join Competition)
make data                  # downloads and unzips the two files below into data/
```

## Manual

1. Log in to Kaggle and open <https://www.kaggle.com/competitions/home-credit-default-risk>, then **accept the rules** (Join Competition).
2. Download from the Data tab and unzip so these two files exist:

```
data/application_train.csv        # ~307K rows, one per loan, TARGET = 1 means payment difficulty
data/previous_application.csv     # ~1.67M rows, NAME_CONTRACT_STATUS = Approved / Refused / Canceled / Unused offer
```

Then `make pipeline`. `previous_application.csv` is optional (without it the funnel is skipped).

## Notes on the data

- `DAYS_BIRTH` is negative days. `DAYS_EMPLOYED = 365243` is a placeholder for "not employed" and is treated as missing.
- `TARGET` exists only for loans that were granted: see *Limitations* in the README.
- Currency units are not specified by the dataset; the rule thresholds are expressed in those same units.
