# DVC Workflow

## Concepts
- Data lives in DagsHub storage. Git stores only small pointer files (`*.dvc`) and `dvc.lock`.
- A pointer file contains the md5 hash of the dataset, so Git commit X always maps to dataset X.

## Daily commands
```bash
git pull && dvc pull      # get the latest code and its matching data
dvc repro                 # re-run pipeline stages whose inputs changed
dvc status                # what changed?
```

## Creating a new dataset version
```bash
python -m src.data.ingestion        # produces a new data/raw/energy_raw.csv
dvc add data/raw/energy_raw.csv     # updates the .dvc pointer
dvc repro                           # re-validate
git add data/raw/energy_raw.csv.dvc dvc.lock
git commit -m "data: <what changed>"
dvc push && git push
```

## Restoring an older version
```bash
git checkout v1-data      # any tag or commit
dvc checkout              # restores the matching dataset from the local cache
# if the data is not in your cache: dvc pull
git checkout main && dvc checkout    # go back
```

## Resolving merge conflicts in `.dvc` files and `dvc.lock`
A conflict means two branches produced different datasets. Never hand-edit the hashes.
1. `git merge <branch>` reports `CONFLICT` in `data/raw/energy_raw.csv.dvc`.
2. Decide which dataset should win, or whether a combined one is needed.
   - Keep one side: `git checkout --ours data/raw/energy_raw.csv.dvc` (or `--theirs`), then `dvc checkout`.
   - Build a merged dataset: regenerate the data (for example, re-run ingestion with the combined date range), then `dvc add data/raw/energy_raw.csv` to rewrite the pointer.
3. For `dvc.lock`, take either side and run `dvc repro` to regenerate it correctly.
4. `git add` the resolved files, `git commit`, then `dvc push`.
5. Conflicts in `dvc.yaml` or `configs/` are ordinary text conflicts: edit and commit as usual.

## Credentials
Each member keeps their own DagsHub token in `.dvc/config.local` (never committed).
