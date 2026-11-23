# Invalid Archived Results (Pre-SCORER_VERSION 2.0.0)

**DO NOT CITE THESE RESULTS.**

All files in this directory (`pilot/`, `depth/`, `efficiency_results.json`) were produced by an earlier version of EpiBench prior to `SCORER_VERSION = "2.0.0"` and are **invalid** for the following verified reasons:

1. **Buggy scorers**:
   - `check_string_match` dropped tokens of length <= 2 (allowing wrong versions such as "PostgreSQL 14" to pass for gold "PostgreSQL 16").
   - Substring matching passed negated answers such as "No longer stands" for gold "Yes, it stands".
   - `check_integer_match` accepted any occurrence of a digit anywhere in prose (including bullet-list numbers like `2.`), passing wrong counts.
   - `check_set_overlap` used hard-coded scenario keywords, a loose 40% keyword threshold, and penalized valid answers mentioning the word "revisit".
2. **Stale generators and ground-truth errors**:
   - 16 of 42 pilot questions do not match what the generator produces.
   - Track F Titan slip days was hardcoded as 52 instead of the true 2024 leap-year value of 53.
   - Track R pool 6 contained duplicate values (`December 15`, `January 15`, `January 15`).
3. **Flawed adapter and metric execution**:
   - No adapter honored `as_of` filtering; `full_context` truncated newest turns instead of oldest; Track E never called `adapter.delete()`; missing dimensions defaulted to 100.0%.

These files are retained solely for historical transparency and auditability.
