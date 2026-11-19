"""Build the released files from the two pipeline outputs.

Inputs (not redistributed here):
  --screening  crypto_classification_final_results.csv  one row per validated BTC address (23,882) with the
               LLM screening result of scripts/04_run_llm_screening.py; label 'illicit' marks the released
               positives (2,438).
  --onchain    address_stats.parquet  on-chain statistics for the same 23,882 addresses.

Outputs (written under --out):
  data/positive_addresses.csv   address, category                        (2,438 rows)
  data/candidates.csv           address, screening result, on-chain stats  (23,882 rows)
  data/screening_outputs.csv    structured LLM outputs, NO forum text       (23,882 rows)
  results/category_counts.csv   category distribution of the positives
  results/summary_metrics.json  counts and on-chain medians, computed from the files above

No forum text, thread title, user name or LLM justification is written: the raw LLM category strings
sometimes echo post text, so categories are kept only when they parse to an exact taxonomy name;
anything else is written as 'Unmapped'.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

# category names returned by the code's prompt -> names used in the paper
ILLICIT = {"abuse": "Abuse", "account selling": "Account Selling", "counterfeit": "Counterfeit",
           "card dumps": "Card Dumps", "card dump": "Card Dumps", "drug": "Drugs", "drugs": "Drugs",
           "fraud": "Fraud", "investment": "Investment", "investments": "Investment",
           "laundering": "Laundering", "malware": "Malware", "membership": "Membership",
           "memberships": "Membership", "service": "Services", "services": "Services",
           "weapon": "Weapons", "weapons": "Weapons", "other": "Other", "others": "Other",
           "unclear": "Unclear"}
LEGIT = {c.lower(): c for c in ("Donation request", "Knowledge sharing", "Escrow verification",
                                "User identification", "Product advertising", "Legal services", "Other")}


def parse_label(raw, table):
    """Exact taxonomy name from a raw LLM answer (plain string or JSON object), else 'Unmapped'."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except ValueError:
            return "Unmapped"
        s = next((obj[k] for k in ("category", "content", "response") if isinstance(obj.get(k), str)), "")
    s = s.strip().strip("*\"'` .").lower()
    return table.get(s, "Unmapped")


def parse_votes(raw):
    """List of vote dicts. Some stored justifications contain raw newlines or were cut mid-string;
    those records fall back to reading only the "vote" / "model" fields with a regex."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        v = json.loads(raw, strict=False)
        return v if isinstance(v, list) else []
    except ValueError:
        pass
    try:
        v = ast.literal_eval(raw)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        pass
    votes = re.findall(r'"vote":\s*"(Yes|No)"', raw)
    models = re.findall(r'"model":\s*"([^"]+)"', raw)
    return [{"vote": x, "model": models[i] if i < len(models) else ""} for i, x in enumerate(votes)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", required=True)
    ap.add_argument("--onchain", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(a.screening)
    oc = pd.read_parquet(a.onchain)
    assert s.address.is_unique and oc.address.is_unique and set(s.address) == set(oc.address), "address sets differ"

    s["illicit_category_std"] = s.illicit_category.map(lambda x: parse_label(x, ILLICIT))
    s["legitimate_category_std"] = s.legitimate_category.map(lambda x: parse_label(x, LEGIT))
    votes = s.llm_votes.map(parse_votes)
    s["n_llm_votes"] = votes.map(len)
    s["llm_models"] = votes.map(lambda v: ";".join(sorted({str(x.get("model", "")) for x in v if isinstance(x, dict)} - {""})))

    # screening outputs (structured fields only)
    # an address was shown to an LLM only if posts were found for it; the source file still labels the
    # others 'legitimate' (no posts found) or 'error'
    s["llm_screened"] = s.n_llm_votes > 0
    so = pd.DataFrame({
        "address": s.address,
        "screening_label": s.label,                  # illicit / legitimate / error (as in the source file)
        "llm_screened": s.llm_screened,
        "n_llm_votes": s.n_llm_votes,
        "yes_votes": s.vote_count,
        "average_confidence": s.average_confidence.round(4),
        "llm_models": s.llm_models,
        "illicit_category": s.illicit_category_std,
        "legitimate_category": s.legitimate_category_std,
        "n_posts_screened": s.supporting_posts_count,
    }).sort_values("address")
    so.to_csv(out / "data/screening_outputs.csv", index=False)

    # positives
    pos = s[s.label == "illicit"][["address", "illicit_category_std"]].rename(columns={"illicit_category_std": "category"})
    pos = pos.sort_values("address")
    pos.to_csv(out / "data/positive_addresses.csv", index=False)

    # candidates = all validated BTC addresses with on-chain statistics
    c = s[["address", "label", "llm_screened"]].rename(columns={"label": "screening_label"}).merge(oc, on="address")
    c["released_positive"] = c.screening_label.eq("illicit")
    cand = c[["address", "screening_label", "llm_screened", "released_positive", "nb_transactions", "total_received",
              "total_sent", "balance", "first_apparition_date", "last_activity_date"]].rename(columns={
        "total_received": "total_received_sat", "total_sent": "total_sent_sat", "balance": "balance_sat",
        "first_apparition_date": "first_onchain_tx", "last_activity_date": "last_onchain_tx"}).sort_values("address")
    cand.to_csv(out / "data/candidates.csv", index=False)

    # results
    cc = pos.category.value_counts().rename_axis("category").reset_index(name="n_addresses")
    cc["share_pct"] = (100 * cc.n_addresses / len(pos)).round(2)
    cc.to_csv(out / "results/category_counts.csv", index=False)

    def med(g):
        return {"n": int(len(g)), "tx_count_median": float(g.nb_transactions.median()),
                "btc_received_median": round(float(g.total_received_sat.median()) / 1e8, 4),
                "btc_sent_median": round(float(g.total_sent_sat.median()) / 1e8, 4),
                "first_onchain_tx": str(g.first_onchain_tx.min()), "last_onchain_tx": str(g.last_onchain_tx.max())}

    summary = {
        "validated_btc_candidates": int(len(cand)),
        "released_positive_addresses": int(len(pos)),
        "screening_label_counts": {k: int(v) for k, v in s.label.value_counts().items()},
        "llm_screened": int(s.llm_screened.sum()),
        "llm_screened_by_models": {k: int(v) for k, v in s[s.llm_screened].llm_models.value_counts().items()},
        "not_screened_no_posts_found": int((~s.llm_screened).sum()),
        "positives_by_n_yes_votes": {str(k): int(v) for k, v in s[s.label == "illicit"].vote_count.value_counts().sort_index().items()},
        "llm_models_seen": sorted({m for x in s.llm_models for m in x.split(";") if m}),
        "onchain": {"all": med(cand), "released_positive": med(cand[cand.released_positive]),
                    "other": med(cand[~cand.released_positive])},
        "category_counts": dict(zip(cc.category, map(int, cc.n_addresses))),
        "inputs_sha256": {Path(a.screening).name: sha256(a.screening), Path(a.onchain).name: sha256(a.onchain)},
    }
    (out / "results/summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("validated_btc_candidates", "released_positive_addresses",
                                              "screening_label_counts", "category_counts")}, indent=1))


if __name__ == "__main__":
    main()
