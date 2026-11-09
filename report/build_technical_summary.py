"""Typeset the existing summary Markdown with the installed pdflatex (no Python extras).

Run from the repository root: python3 report/build_technical_summary.py
The Markdown is the content source; intermediate TeX/log files stay in tmp/pdfs/.
"""
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report/technical_summary_draft.md"
WORK = ROOT / "tmp/pdfs/technical_summary"
OUTPUT = ROOT / "report/technical_summary_P22.pdf"


def escape(text):
    symbols = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "×": r"\ensuremath{\times}", "—": "---", "–": "--", "·": r"\textperiodcentered{}",
    }
    return "".join(symbols.get(c, c) for c in text)


def inline(text):
    tokens = re.split(r"(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))", text)
    rendered = []
    for token in tokens:
        if token.startswith("**"):
            rendered.append(r"\textbf{" + inline(token[2:-2]) + "}")
        elif token.startswith("`"):
            rendered.append(r"\texttt{" + escape(token[1:-1]) + "}")
        elif (match := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)):
            # Repository-relative figure references remain meaningful in the standalone PDF.
            url = match[2]
            if url.startswith("../"):
                url = "https://github.com/1r444444/CV-P22-building-segmentation/blob/main/" + url[3:]
            rendered.append(r"\href{" + escape(url) + "}{" + escape(match[1]) + "}")
        else:
            rendered.append(escape(token))
    return "".join(rendered)


def render(source):
    source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
    blocks = re.split(r"\n\s*\n", source.strip())
    output = []
    for block in blocks:
        lines = block.splitlines()
        if block.startswith("# "):
            output.append(r"{\LARGE\bfseries " + inline(block[2:]) + r"\par}\vspace{3pt}")
        elif block.startswith("## "):
            if block.startswith("## 6."):
                output.append(r"\newpage")
            output.append(r"\section*{" + inline(block[3:]) + "}")
        elif block.startswith("| "):
            rows = [line.strip().strip("|").split("|") for line in lines]
            wrapped = len(rows[0]) == 2
            table = r"\begin{tabularx}{\linewidth}{@{}p{4.2cm}X@{}}" if wrapped else r"\begin{tabular}{@{}lrrrr@{}}"
            output.extend([r"\begin{center}", table, r"\toprule"])
            for index, row in enumerate(rows):
                if index == 1:
                    output.append(r"\midrule")
                    continue
                cells = [inline(cell.strip()) for cell in row]
                if index == 0:
                    cells = [r"\textbf{" + cell + "}" for cell in cells]
                output.append(" & ".join(cells) + r" \\")
            end_table = r"\end{tabularx}" if wrapped else r"\end{tabular}"
            output.extend([r"\bottomrule", end_table, r"\end{center}"])
        elif block.startswith("- "):
            output.append(r"\begin{itemize}")
            for item in re.split(r"\n- ", block[2:]):
                output.append(r"\item " + inline(" ".join(item.splitlines())))
            output.append(r"\end{itemize}")
        else:
            paragraph = " ".join(lines)
            # Make the repository URL clickable and avoid a long unbreakable line.
            repo = "https://github.com/1r444444/CV-P22-building-segmentation"
            if paragraph == "Repository: " + repo + ".":
                output.append(r"Repository: \url{" + repo + "}.")
            else:
                output.append(inline(paragraph) + "\n")
    return "\n".join(output)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,enumitem,tabularx}
\usepackage[top=1.5cm,bottom=1.5cm,left=1.8cm,right=1.8cm]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\hypersetup{pdftitle={P22 - Building Segmentation in Satellite Images},
pdfauthor={Renata Salikhzyanova; Kristina Soloveva; Elena Tuturina; Irina Bychkova}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlength{\emergencystretch}{3em}
\setlist[itemize]{leftmargin=12pt,itemsep=2pt,parsep=0pt,topsep=3pt}
\makeatletter
\renewcommand\section{\@startsection{section}{1}{0pt}{1.2ex plus .2ex}{.4ex}{\normalfont\normalsize\bfseries}}
\makeatother
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\small P22 | Computer Vision 2026}
\fancyfoot[R]{\small \thepage/2}
\setlength{\footskip}{18pt}
\begin{document}
"""


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    tex = WORK / "technical_summary_P22.tex"
    tex.write_text(PREAMBLE + render(SOURCE.read_text()) + "\n\\end{document}\n")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=WORK, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    pdf = WORK / "technical_summary_P22.pdf"
    # Never publish a build with an accidental third page.
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    if not re.search(r"^Pages:\s+2\s*$", info, flags=re.M):
        raise RuntimeError("Expected exactly two pages; inspect tmp/pdfs/technical_summary/.")
    shutil.copyfile(pdf, OUTPUT)
    print(f"Created {OUTPUT} (2 pages). Render and visually inspect both pages before submission.")
