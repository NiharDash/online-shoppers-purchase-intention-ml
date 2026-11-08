"""
Build the single-PDF submission for ML Assignment 2.

Assembles, in the order the brief requires:
  1. GitHub repository link
  2. Live Streamlit app link
  3. Screenshot of the assignment being executed
  4. The full README.md content

Usage:
    python build_submission_pdf.py
    python build_submission_pdf.py --bits-id 2023AB12345 \
        --run-screenshot /path/to/lab_screenshot.png

Requires: markdown, and Google Chrome (used headlessly to render the PDF).
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent

GITHUB_URL = "https://github.com/NiharDash/online-shoppers-purchase-intention-ml"
STREAMLIT_URL = "https://nihar-online-shoppers.streamlit.app"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def embed_image(path: Path | None) -> str:
    """Inline an image as a data URI so the HTML is self-contained."""
    if not path or not path.exists():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def screenshot_block(path: Path | None, caption: str, placeholder: str) -> str:
    src = embed_image(path)
    if not src:
        return (
            f'<div class="placeholder"><b>Screenshot not supplied.</b><br>{placeholder}</div>'
        )
    return (
        f'<figure><img src="{src}" alt="{html.escape(caption)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def build_html(args) -> str:
    readme_md = (ROOT / "README.md").read_text()
    readme_html = markdown.markdown(
        readme_md, extensions=["tables", "fenced_code", "sane_lists"]
    )

    run_shot = screenshot_block(
        Path(args.run_screenshot) if args.run_screenshot else None,
        "Execution of <code>train_models.py</code> — all five classifiers trained "
        "and evaluated on the held-out test split.",
        "Re-run this script with <code>--run-screenshot &lt;path&gt;</code>.",
    )
    app_shot = screenshot_block(
        Path(args.app_screenshot) if args.app_screenshot else None,
        "The deployed Streamlit application, showing metrics, confusion matrix "
        "and ROC curve for the selected model.",
        "Optional.",
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ML Assignment 2 — Submission</title>
<style>
  @page {{ size: A4; margin: 15mm 15mm 17mm 15mm; }}
  body {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt;
         line-height: 1.55; color: #111; margin: 0; }}

  .cover {{ text-align: center; padding-top: 42mm; page-break-after: always; }}
  .cover h1 {{ font-size: 23pt; margin: 0 0 6px; color: #1f4e79; }}
  .cover h2 {{ font-size: 13pt; font-weight: normal; color: #444; margin: 0 0 30px; }}
  .cover .meta {{ display: inline-block; text-align: left; border: 1px solid #aaa;
                 border-radius: 7px; padding: 16px 34px; font-size: 11pt; }}
  .cover .meta div {{ margin: 5px 0; }}
  .cover .meta b {{ display: inline-block; width: 132px; color: #1f4e79; }}

  h2.sec {{ font-size: 13pt; color: #fff; background: #1f4e79; padding: 6px 13px;
           border-radius: 4px; margin: 20px 0 11px; page-break-after: avoid; }}

  .linkbox {{ border: 2px solid #1f4e79; border-radius: 7px; padding: 13px 18px;
             margin: 10px 0 16px; background: #f4f8fc; page-break-inside: avoid; }}
  .linkbox .lbl {{ font-size: 9pt; color: #666; text-transform: uppercase;
                  letter-spacing: .7px; }}
  .linkbox a {{ font-size: 12.5pt; color: #1565c0; word-break: break-all;
               text-decoration: none; font-weight: bold; }}

  figure {{ margin: 10px 0 14px; page-break-inside: avoid; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #bbb; border-radius: 5px; }}
  figcaption {{ font-size: 9pt; color: #555; margin-top: 6px; font-style: italic; }}

  .placeholder {{ border: 2px dashed #c9a227; background: #fdf8e8; border-radius: 7px;
                 padding: 22px; text-align: center; font-size: 10pt; color: #7a5f10; }}

  .note {{ border-left: 4px solid #c9a227; background: #fdf8e8; padding: 8px 13px;
          font-size: 9.5pt; margin: 10px 0; }}

  .readme {{ page-break-before: always; }}
  .readme h1 {{ font-size: 16pt; color: #1f4e79; border-bottom: 2px solid #1f4e79;
               padding-bottom: 5px; }}
  .readme h2 {{ font-size: 12.5pt; color: #1f4e79; margin-top: 18px;
               border-bottom: 1px solid #ccd; padding-bottom: 3px; page-break-after: avoid; }}
  .readme h3 {{ font-size: 11pt; color: #2d4f8a; margin-top: 13px; page-break-after: avoid; }}
  .readme table {{ border-collapse: collapse; width: 100%; margin: 9px 0;
                  font-size: 8.6pt; page-break-inside: avoid; }}
  .readme th, .readme td {{ border: 1px solid #8fa6c8; padding: 4px 6px;
                           text-align: left; vertical-align: top; }}
  .readme th {{ background: #dce5f5; color: #1f4e79; }}
  .readme code {{ background: #eef1f6; padding: 1px 4px; border-radius: 3px;
                 font-family: Menlo, Consolas, monospace; font-size: 8.8pt; }}
  .readme pre {{ background: #1e1e2e; color: #d4d4e4; padding: 10px 13px;
                border-radius: 5px; font-size: 8.6pt; overflow-x: auto;
                page-break-inside: avoid; }}
  .readme pre code {{ background: none; color: inherit; padding: 0; }}
  .readme blockquote {{ border-left: 4px solid #c9a227; background: #fdf8e8;
                       margin: 9px 0; padding: 7px 13px; font-size: 9.6pt; }}
  .readme hr {{ border: none; border-top: 1px solid #ccd; margin: 16px 0; }}
  .readme a {{ color: #1565c0; word-break: break-all; }}
  .readme li {{ margin-bottom: 3px; }}
</style></head>
<body>

<div class="cover">
  <h1>Machine Learning — Assignment 2</h1>
  <h2>M.Tech (AIML) · Work Integrated Learning Programmes Division</h2>
  <div class="meta">
    <div><b>Name</b> {html.escape(args.name)}</div>
    <div><b>BITS ID</b> {html.escape(args.bits_id)}</div>
    <div><b>Course</b> Machine Learning</div>
    <div><b>Course No.</b> S1-26_AIMLCZC418</div>
    <div><b>Dataset</b> Online Shoppers Purchasing Intention (UCI)</div>
    <div><b>Submitted</b> {html.escape(args.date)}</div>
  </div>
</div>

<h2 class="sec">1 · GitHub Repository Link</h2>
<div class="linkbox">
  <div class="lbl">Repository</div>
  <a href="{GITHUB_URL}">{GITHUB_URL}</a>
</div>
<p>The repository contains the complete source code, <code>requirements.txt</code>,
<code>README.md</code>, the test data used in the experiments
(<code>test_data.csv</code>), and a <code>model/</code> directory holding all five
trained classifiers serialised as <code>.joblib</code> files.</p>

<h2 class="sec">2 · Live Streamlit App Link</h2>
<div class="linkbox">
  <div class="lbl">Deployed on Streamlit Community Cloud</div>
  <a href="{STREAMLIT_URL}">{STREAMLIT_URL}</a>
</div>
<p>The app opens directly on the bundled test split, so an interactive frontend with
populated results appears immediately. It supports CSV upload, model selection,
evaluation-metric display, and a confusion matrix plus classification report.</p>
{app_shot}

<h2 class="sec">3 · Screenshot of Assignment Execution</h2>
{run_shot}
{args.screenshot_note}

<div class="readme">
<h2 class="sec">4 · README.md Content</h2>
{readme_html}
</div>

</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Nihar Dash")
    ap.add_argument("--bits-id", default="____________________")
    ap.add_argument("--date", default="08 November 2026")
    ap.add_argument("--run-screenshot", default="/tmp/run_screenshot.png")
    ap.add_argument("--app-screenshot", default="/tmp/cloud.png")
    ap.add_argument("--screenshot-note", default="")
    ap.add_argument("--out", default=str(ROOT / "ML_Assignment2_Submission.pdf"))
    args = ap.parse_args()

    html_path = ROOT / "_submission.html"
    html_path.write_text(build_html(args))

    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}; cannot render PDF.")

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={args.out}", html_path.as_uri()],
        check=True, capture_output=True,
    )
    html_path.unlink()
    size = Path(args.out).stat().st_size
    print(f"Wrote {args.out}  ({size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
