# hackforums-bitcoin-labels

Bitcoin addresses shared on HackForums (CrimeBB archive), validated on-chain and screened with a local LLM
for illicit use. The release contains 23,882 validated Bitcoin addresses and 2,438 addresses labelled illicit,
with a cybercrime category for each. No forum text, thread titles or user identifiers are included
(see [DATA_STATEMENT.md](DATA_STATEMENT.md)).

## Contents

| Path | What it is |
|---|---|
| `data/positive_addresses.csv` | 2,438 illicit addresses: `address`, `category` |
| `data/candidates.csv` | 23,882 validated BTC addresses: screening result and on-chain statistics |
| `data/screening_outputs.csv` | Structured LLM screening outputs per address (votes, confidence, category); no forum text |
| `config/address_regex.yaml` | Address regular expressions |
| `config/pipeline.yaml` | Validation, context and LLM settings as used in the code |
| `prompts/illicit_address_screening.txt` | The exact prompt strings used by the screening script |
| `scripts/01_extract_addresses.py` | Extracts BTC addresses from posts and contracts and validates them on-chain (steps 1 and 2) |
| `scripts/04_run_llm_screening.py` | Builds each address's context and runs the LLM screening (steps 3 and 4) |
| `scripts/08_build_release.py` | Builds every file in `data/` and `results/` from the pipeline outputs |
| `results/category_counts.csv` | Category distribution of the 2,438 positives |
| `results/summary_metrics.json` | Counts and on-chain medians computed from `data/` |

## Data dictionary

`data/candidates.csv`
- `screening_label`: `illicit`, `legitimate` or `error`, as produced by the screening step.
- `llm_screened`: `False` when no forum post was retrieved for the address, so it was never sent to the LLM
  (5,354 addresses; 5,242 of them carry `screening_label = legitimate`, 112 carry `error`).
- `released_positive`: the address is in `positive_addresses.csv`.
- `nb_transactions`, `total_received_sat`, `total_sent_sat`, `balance_sat` (satoshi), `first_onchain_tx`,
  `last_onchain_tx` (UTC; latest activity in the file is 2024-09-24).

`data/screening_outputs.csv`
- `n_llm_votes`, `yes_votes`: number of models that answered, and how many answered Yes to the illicit prompt.
  An address is `illicit` when `yes_votes >= 1`.
- `llm_models`: `qwen3:30b` alone (9,004 addresses) or `qwen3:30b` and `deepseek-r1:70b` (9,524 addresses).
- `illicit_category` / `legitimate_category`: the model's answer when it parses exactly to a category name;
  any other answer is written as `Unmapped`. The raw answers are not released because some of them echo post text.

## Rebuilding the release

The pipeline needs a local copy of the CrimeBB HackForums database, which is available only under the
Cambridge Cybercrime Centre data use agreement. With the two pipeline outputs available locally:

```bash
pip install -r requirements.txt
python scripts/08_build_release.py --screening crypto_classification_final_results.csv \
    --onchain address_stats.parquet --out .
```

## Not included yet

- `data/expert_review.csv`: the per-address expert decisions for the LLM-flagged candidates.
- The audit of a random sample of 200 unflagged addresses (`scripts/06_sample_unflagged_audit.py`).
- `scripts/05_aggregate_screening.py` and `scripts/07_compute_metrics.py`.
- Forum-side metadata for each address (first/last forum appearance, number of mentions).
- Validation code for Litecoin, Ethereum and Dash addresses.
- `LICENSE`.
