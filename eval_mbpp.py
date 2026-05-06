"""Evaluate the Kimi-backed coding agent on the first 50 MBPP sanitized problems.

Each problem is generated once; if it fails, a single self-correction retry is
attempted and recorded in the `corrected_code` / `corrected_result` columns.
"""

import csv
from datasets import load_dataset

from agent import generate_code, execute_code, fix_code


OUTPUT_CSV = "mbpp_results.csv"
NUM_PROBLEMS = 50

FIELDNAMES = [
    "task_id", "text", "generated_code",
    "stdout", "stderr", "returncode", "result",
    "corrected_code", "corrected_result",
]


def classify(stdout: str, stderr: str, returncode: int) -> str:
    """Map an execution result to one of the fixed result labels."""
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


def run_eval(num_problems: int = NUM_PROBLEMS, output_csv: str = OUTPUT_CSV,
             progress_cb=None):
    """Run the MBPP eval. `progress_cb(i, total, task_id)` is called before each problem."""
    ds = load_dataset("mbpp", "sanitized", split="test")
    rows = []
    total = min(num_problems, len(ds))

    for i in range(total):
        problem = ds[i]
        task_id = problem["task_id"]
        text = problem.get("text") or problem.get("prompt")
        tests = problem["test_list"]
        tests_block = "\n".join(tests)

        if progress_cb is not None:
            progress_cb(i, total, task_id)
        else:
            print(f"[{i+1}/{total}] task_id={task_id}")

        try:
            generated = generate_code(text)
        except Exception as e:
            rows.append({
                "task_id": task_id, "text": text, "generated_code": "",
                "stdout": "", "stderr": f"generation_error: {e}",
                "returncode": -2, "result": "other_error",
                "corrected_code": "", "corrected_result": "other_error",
            })
            continue

        full_code = generated + "\n\n" + tests_block + "\n"
        stdout, stderr, returncode = execute_code(full_code)
        result = classify(stdout, stderr, returncode)

        if result == "pass":
            corrected_code = ""
            corrected_result = "skipped"
        else:
            try:
                corrected_code = fix_code(text, generated, stderr)
                corrected_full = corrected_code + "\n\n" + tests_block + "\n"
                c_stdout, c_stderr, c_returncode = execute_code(corrected_full)
                corrected_result = classify(c_stdout, c_stderr, c_returncode)
            except Exception as e:
                corrected_code = ""
                corrected_result = "other_error"
                print(f"  fix_code error: {e}")

        rows.append({
            "task_id": task_id,
            "text": text,
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
