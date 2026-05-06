# Autonomous Coding Agents — Kimi K2.6

A small, transparent self-correcting coding agent built around Moonshot's
Kimi K2.6 frontier model, paired with a Streamlit dashboard that makes
every failure inspectable.

The goal of the project is **not** to beat a benchmark — it is to learn
what it takes to wire up a frontier model end-to-end, run its code
safely, and expose its mistakes in a UI so we can characterize where and
why the model struggles.

---

## What's in here

Three layers, from the model up to the UI.

### 1. The agent — `agent.py`
Talks to Kimi K2.6 over the OpenAI-compatible Moonshot endpoint.

- `generate_code(task)` — natural-language task → Python code.
- `execute_code(code)` — runs the code in a `subprocess` sandbox with a
  10-second timeout, so runaway or crashing code can't affect the agent.
- `fix_code(task, code, error)` — one self-correction pass: original task
  + failing code + stderr → corrected code.
- `main()` — REPL: `task> ...`.

### 2. The evaluation harness — three datasets, one pipeline
Each script writes the same CSV schema so the Streamlit renderer is shared.

| Script | Dataset | Purpose |
|---|---|---|
| `eval_mbpp.py` | MBPP sanitized (50) | Natural-language tasks with hidden tests — exposes *spec ambiguity*. |
| `eval_humaneval.py` | HumanEval (164) | Signatures + docstrings given — removes naming noise to test *core reasoning*. |
| `eval_humanevalplus.py` | HumanEval+ (164) | Same problems, ~80× more tests — exposes *edge-case bugs* HumanEval misses. |

Every problem records: prompt, generated code, stdout, stderr, first-attempt
result, corrected code, corrected result. The self-correction loop runs at
most one retry per problem.

### 3. The UI — `app.py` (Streamlit)
Four tabs:
1. **Run Agent** — live REPL: type a task, see the generated code and output.
2. **MBPP Eval Results**
3. **HumanEval Eval Results**
4. **HumanEval+ Eval Results**

Each eval tab shows pass-rate metrics, first-vs-corrected breakdowns,
per-category rates, and a **per-problem inspector** — pick any task, see
the prompt, the failing code, the stderr, and the fix. This is the
transparency piece: every claim about the model is one click away from
the evidence.

The CLI REPL still works (`python agent.py`), but the Streamlit app is
the headline interface.


## Setup

Python 3.14+ recommended (matches `pyproject.toml`). With pip:

```
pip install -r requirements.txt
```

Or with uv (lockfile is checked in):

```
uv sync
```

Then create `.env.local` in the project root with your Moonshot key:

```
# .env.local
kimi_api=sk-...
```

Endpoint: `https://api.moonshot.ai/v1` · Model: `kimi-k2.6` · Temperature: `1`.

### Datasets

All three benchmarks are pulled at runtime by `datasets.load_dataset(...)` and
cached locally — no manual download is required. The first run of each eval
script will download the data:

- MBPP — `mbpp` config `sanitized`, split `test` (Hugging Face). A copy of the
  raw JSON is also bundled at `sanitized-mbpp.json` for offline reference.
- HumanEval — `openai_humaneval`, split `test`.
- HumanEval+ — `evalplus/humanevalplus`.

## Run

```
python agent.py                # CLI REPL
python eval_mbpp.py            # → mbpp_results.csv
python eval_humaneval.py       # → humaneval_results.csv
python eval_humanevalplus.py   # → humanevalplus_results.csv
streamlit run app.py           # full UI (all three benchmarks)
```

## Reproducibility

The numbers reported in this README and in the accompanying report come from
a **single run** of each evaluation script. The setup is intentionally not
deterministic:

- `temperature = 1` is used for both `generate_code` and `fix_code`.
- **No `seed` is passed** to the Moonshot API.
- The Moonshot endpoint does not currently guarantee deterministic outputs,
  so re-running the scripts is expected to produce slightly different
  per-problem outcomes.

For this reason, the **canonical artifacts of the reported run are the CSVs
checked into the repo**:

- `mbpp_results.csv`
- `humaneval_results.csv`
- `humanevalplus_results.csv`

These contain the exact prompts, generated code, stderr, first-attempt
result, corrected code, and corrected result for every problem behind the
headline numbers above. The Streamlit app reads these files directly, so a
grader can inspect every claim without re-running the API. Re-running the
`eval_*.py` scripts will overwrite these CSVs with a fresh sample from the
same distribution, not reproduce them exactly.
