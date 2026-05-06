"""Streamlit UI: run the Kimi coding agent and browse MBPP / HumanEval eval results."""

import os

import pandas as pd
import streamlit as st

from agent import generate_code, execute_code
from eval_mbpp import run_eval as run_mbpp_eval, NUM_PROBLEMS as MBPP_DEFAULT_N
from eval_humaneval import run_eval as run_humaneval_eval, NUM_PROBLEMS as HE_DEFAULT_N
from eval_humanevalplus import run_eval as run_hep_eval, NUM_PROBLEMS as HEP_DEFAULT_N

MBPP_CSV = "mbpp_results.csv"
HUMANEVAL_CSV = "humaneval_results.csv"
HUMANEVALPLUS_CSV = "humanevalplus_results.csv"

CATEGORY_KEYWORDS = {
    "string": ["string", "char", "substring", "word"],
    "list":   ["list", "array", "tuple", "element"],
    "math":   ["sum", "product", "number", "integer", "prime", "factor",
               "square", "power", "divisor", "fibonacci", "math"],
    "sort":   ["sort", "sorted", "order", "ascending", "descending"],
    "file":   ["file", "read ", "write "],
    "dict":   ["dict", "dictionary", "key", "value"],
    "regex":  ["regex", "pattern", "match"],
}


def categorize(text: str) -> list[str]:
    t = text.lower()
    cats = [c for c, kws in CATEGORY_KEYWORDS.items() if any(k in t for k in kws)]
    return cats or ["other"]


st.set_page_config(page_title="Kimi Coding Agent", layout="wide")


def render_eval_panel(label: str, csv_path: str, run_eval_fn,
                      default_n: int, max_n: int, key_prefix: str):
    """Renders the run-control + results panel for one dataset."""
    st.header(f"{label} Eval Results")

    with st.expander(f"Run {label} eval", expanded=not os.path.exists(csv_path)):
        num_problems = st.number_input(
            "Number of problems", min_value=1, max_value=max_n,
            value=default_n, step=1, key=f"{key_prefix}_n",
        )
        if st.button("Run eval", type="primary", key=f"{key_prefix}_run"):
            progress = st.progress(0.0, text="Starting...")
            status = st.empty()

            def cb(i, total, task_id):
                progress.progress(i / total,
                                  text=f"[{i+1}/{total}] task_id={task_id}")

            with st.spinner(f"Running {label} eval — this can take a while..."):
                summary = run_eval_fn(num_problems=int(num_problems),
                                      output_csv=csv_path,
                                      progress_cb=cb)
            progress.progress(1.0, text="Done")
            status.success(
                f"First attempt: {summary['passed']}/{summary['total']} passed. "
                f"Recovered after fix: {summary['recovered']}/{summary['failed']}."
            )
            st.rerun()

    if not os.path.exists(csv_path):
        st.info(f"{csv_path} not found. Run the eval above (or `python {key_prefix}_eval.py`).")
        return

    df = pd.read_csv(csv_path).fillna("")
    total = len(df)
    passed = int((df["result"] == "pass").sum())

    has_correction = "corrected_result" in df.columns
    if has_correction:
        skipped = int((df["corrected_result"] == "skipped").sum())
        first_fail = total - skipped
        recovered = int(
            ((df["result"] != "pass") & (df["corrected_result"] == "pass")).sum()
        )
        final_pass = passed + recovered

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("First-attempt pass",
                  f"{passed}/{total}",
                  f"{100*passed/total:.1f}%" if total else "—")
        c3.metric("Recovered after fix",
                  f"{recovered}/{first_fail}" if first_fail else "0/0",
                  f"{100*recovered/first_fail:.1f}%" if first_fail else "—")
        c4.metric("Final pass",
                  f"{final_pass}/{total}",
                  f"{100*final_pass/total:.1f}%" if total else "—")

        st.subheader("First-attempt vs. after-correction pass rate")
        pass_df = pd.DataFrame({
            "stage": ["first attempt", "after correction"],
            "pass rate (%)": [
                100 * passed / total if total else 0,
                100 * final_pass / total if total else 0,
            ],
        })
        st.bar_chart(pass_df, x="stage", y="pass rate (%)")
        st.caption(f"{skipped} problems skipped correction "
                   "(first attempt already passed).")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Passed", passed)
        c3.metric("Pass rate", f"{100*passed/total:.1f}%" if total else "—")

    st.subheader("Result label breakdown")
    if has_correction:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("First attempt")
            st.bar_chart(df["result"].value_counts())
        with col_b:
            st.caption("After correction")
            st.bar_chart(df["corrected_result"].value_counts())
    else:
        st.bar_chart(df["result"].value_counts())

    st.subheader("Per-category pass rate")
    cat_rows = []
    for cat in list(CATEGORY_KEYWORDS.keys()) + ["other"]:
        matching = df[df["text"].apply(lambda t: cat in categorize(t))]
        n = len(matching)
        if n == 0:
            continue
        first_p = int((matching["result"] == "pass").sum())
        if has_correction:
            final_p = first_p + int(
                ((matching["result"] != "pass") &
                 (matching["corrected_result"] == "pass")).sum()
            )
        else:
            final_p = first_p
        cat_rows.append({
            "category": cat,
            "n": n,
            "first-attempt pass rate": f"{100*first_p/n:.1f}%",
            "final pass rate": f"{100*final_p/n:.1f}%",
        })
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True,
                 hide_index=True)

    st.subheader("Per-problem results")
    display_cols = ["task_id", "result"]
    if has_correction:
        display_cols.append("corrected_result")
    display_cols.append("text")
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    with st.expander("Inspect a problem"):
        ids = df["task_id"].tolist()
        choice = st.selectbox("task_id", ids, key=f"{key_prefix}_inspect")
        row = df[df["task_id"] == choice].iloc[0]
        st.markdown(f"**Prompt:**")
        st.code(row["text"], language="python" if str(row["text"]).lstrip().startswith("def ") else "text")
        st.markdown(f"**Result:** `{row['result']}`")
        st.code(row["generated_code"], language="python")
        if isinstance(row["stderr"], str) and row["stderr"].strip():
            st.markdown("**stderr:**")
            st.code(row["stderr"], language="text")
        if has_correction and str(row.get("corrected_code", "")).strip():
            st.markdown(f"**Corrected result:** `{row['corrected_result']}`")
            st.code(row["corrected_code"], language="python")


tab_run, tab_mbpp, tab_he, tab_hep = st.tabs(
    ["Run Agent", "MBPP Eval Results", "HumanEval Eval Results",
     "HumanEval+ Eval Results"]
)

with tab_run:
    st.header("Coding Agent")
    task = st.text_area("Task", placeholder="e.g. Write a function to reverse a string.",
                        height=120)
    if st.button("Run", type="primary", disabled=not task.strip()):
        with st.spinner("Generating..."):
            code = generate_code(task)
        st.subheader("Generated code")
        st.code(code, language="python")

        with st.spinner("Executing..."):
            stdout, stderr, returncode = execute_code(code)

        if returncode == 0:
            st.subheader("Output")
            st.code(stdout or "(no output)", language="text")
        else:
            st.subheader("Error")
            st.code(stderr, language="text")

with tab_mbpp:
    render_eval_panel("MBPP", MBPP_CSV, run_mbpp_eval,
                      MBPP_DEFAULT_N, max_n=500, key_prefix="mbpp")

with tab_he:
    render_eval_panel("HumanEval", HUMANEVAL_CSV, run_humaneval_eval,
                      HE_DEFAULT_N, max_n=164, key_prefix="humaneval")

with tab_hep:
    render_eval_panel("HumanEval+", HUMANEVALPLUS_CSV, run_hep_eval,
                      HEP_DEFAULT_N, max_n=164, key_prefix="humanevalplus")
