"""Build the manuscript twice, verify companions, and inspect the final log.

Run with Python from any directory. Requires pdflatex on PATH. Build artifacts
stay in the repository's ignored build/research directory. No shell escape.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    directory = Path(__file__).resolve().parent
    source = directory / "legibility_optimizer_ud5.tex"
    text = source.read_text(encoding="utf-8")
    listings = re.findall(r"\\lstinputlisting(?:\s*\[[\s\S]*?\])?\s*\{([^}]+)\}", text)
    missing = [name for name in listings if not (directory / name).is_file()]
    if missing:
        raise SystemExit("Missing listing companions: " + ", ".join(missing))
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise SystemExit("pdflatex unavailable; manuscript has NOT been compiled")
    output = directory.parents[1] / "build" / "research"
    output.mkdir(parents=True, exist_ok=True)
    for pass_number in (1, 2):
        result = subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
             "-no-shell-escape", "-output-directory=" + str(output), source.name],
            cwd=directory, capture_output=True, text=True, errors="replace", timeout=180)
        transcript = output / f"pdflatex-pass-{pass_number}.txt"
        transcript.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            print((result.stdout + result.stderr)[-6000:])
            raise SystemExit(f"LaTeX pass {pass_number} failed; inspect {transcript}")
    log = (output / source.with_suffix(".log").name).read_text(encoding="utf-8", errors="replace")
    problems = [line for line in log.splitlines()
                if any(pattern in line for pattern in
                       ("LaTeX Warning", "Package hyperref Warning", "Overfull", "Underfull",
                        "Missing character", "Undefined control sequence"))]
    print(f"Verified {len(listings)} listing files; completed two pdflatex passes.")
    for problem in problems:
        print(problem)
    if any("undefined" in line.lower() or "Rerun" in line for line in problems):
        raise SystemExit("Unresolved references remain; inspect the build log")
    print("PDF: " + str(output / source.with_suffix(".pdf").name))
    print("Log: " + str(output / source.with_suffix(".log").name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
