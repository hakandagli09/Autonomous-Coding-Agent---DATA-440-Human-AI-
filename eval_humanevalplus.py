"""Evaluate the Kimi-backed coding agent on HumanEval+ (EvalPlus).

HumanEval+ keeps the same 164 problems as HumanEval but ships ~80x more tests
per problem, exposing edge cases HumanEval misses. Schema and harness are
identical to HumanEval, so this file is a near-clone of `eval_humaneval.py`
that only swaps the dataset and output CSV.
"""

import csv
from datasets import load_dataset

from agent import generate_code, execute_code, fix_code


OUTPUT_CSV = "humanevalplus_results.csv"
NUM_PROBLEMS = 50

FIELDNAMES = [
    "task_id", "text", "generated_code",
    "stdout", "stderr", "returncode", "result",
    "corrected_code", "corrected_result",
]


def classify(stdout: str, stderr: str, returncode: int) -> str:
    if returncode == 0:
        return "pass"
    if returncode == -1 and "timed out" in stderr.lower():
        return "timeout"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "assertion_error"
    if "Traceback" in stderr or "Error" in stderr:
        return "runtime_error"
    return "other_error"


def build_task_prompt(prompt: str) -> str:
    return (
        "Complete the following Python function. Return the full function "
        "(signature and body) in a single Python code block, with any imports "
        "it needs. Do not include tests or example calls.\n\n"
        f"{prompt}"
    )


def run_eval(num_problems: int = NUM_PROBLEMS, output_csv: str = OUTPUT_CSV,
             progress_cb=None):
    ds = load_dataset("evalplus/humanevalplus", split="test")
    rows = []
    total = min(num_problems, len(ds))

    for i in range(total):
        problem = ds[i]
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]
        task_prompt = build_task_prompt(prompt)
        check_block = f"\n{test}\n\ncheck({entry_point})\n"

        if progress_cb is not None:
            progress_cb(i, total, task_id)
        else:
            print(f"[{i+1}/{total}] task_id={task_id}")

        try:
            generated = generate_code(task_prompt)
        except Exception as e:
            rows.append({
                "task_id": task_id, "text": prompt, "generated_code": "",
                "stdout": "", "stderr": f"generation_error: {e}",
                "returncode": -2, "result": "other_error",
                "corrected_code": "", "corrected_result": "other_error",
            })
            continue

        full_code = generated + check_block
        stdout, stderr, returncode = execute_code(full_code)
        result = classify(stdout, stderr, returncode)

        if result == "pass":
            corrected_code = ""
            corrected_result = "skipped"
        else:
            try:
                corrected_code = fix_code(task_prompt, generated, stderr)
                corrected_full = corrected_code + check_block
                c_stdout, c_stderr, c_returncode = execute_code(corrected_full)
                corrected_result = classify(c_stdout, c_stderr, c_returncode)
            except Exception as e:
                corrected_code = ""
                corrected_result = "other_error"
                print(f"  fix_code error: {e}")

        rows.append({
            "task_id": task_id,
            "text": prompt,
            "generated_code": generated,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "result": result,
            "corrected_code": corrected_code,
            "corrected_result": corrected_result,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for r in rows if r["result"] == "pass")
    failed = len(rows) - passed
    recovered = sum(
        1 for r in rows
        if r["result"] != "pass" and r["corrected_result"] == "pass"
    )
    return {
        "rows": rows,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "recovered": recovered,
        "output_csv": output_csv,
    }


def main():
    summary = run_eval()
    print(f"\nDone. First attempt: {summary['passed']}/{summary['total']} passed.")
    print(f"Recovered after fix: {summary['recovered']}/{summary['failed']} failed problems.")
    print(f"Wrote {summary['output_csv']}")


if __name__ == "__main__":
    main()
