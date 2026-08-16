#!/usr/bin/env python
"""
Generate literate documentation from the examples already in the repository.

Every runnable example in this repo is a pytest function.  This script turns
each one into a documentation page that reads top-to-bottom as prose:

  * comments and docstrings inside the example become the narrative text,
  * the code between them becomes syntax-highlighted blocks,
  * each figure is embedded **at the point in the code that saves it**,
  * an "Imports from this repo" panel names the layer each import comes from
    (lib / model / params), so the dependency structure is visible on the page.

Nothing is authored twice: the docs are a projection of the test suite, so they
cannot drift from it.

Usage
-----
    python docs/generate_docs.py                # run examples, then build pages
    python docs/generate_docs.py --skip-tests   # reuse existing tresults/ output
    python -m mkdocs serve -f docs/mkdocs.yml   # preview

Derived from the earlier generator on the ``refactor/intsqz`` branch, rewritten
for literate output (interleaved prose/code/figures rather than one collapsed
source dump).
"""
import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = Path(__file__).resolve().parent
SITE_SRC = DOCS / "docs"
IMG_DIR = SITE_SRC / "img"

# Trust levels, as characterised by the maintainers.  Surfaced on each page so
# a reader knows how much review the code they are reading has had.
TRUST = {
    "reviewed": (
        "Reviewed",
        "Library code, reviewed by multiple people. Treat as dependable.",
    ),
    "reference": (
        "Reference implementation",
        "The best-known-good state of the internal squeezing model. "
        "Changes here should be checked against the numerical baselines.",
    ),
    "single-author": (
        "Single author",
        "Written by one person; only the simpler cases are well tested. "
        "Verify before relying on results.",
    ),
}

# Which layer each importable module belongs to.  Used for the per-page
# dependency panel and, more importantly, as the machine-readable statement of
# the layering the refactor is meant to make obvious.
LAYERS = {
    "sflu_components.lib": "lib",
    "sflu_components.edges": "lib",
    "sflu_components.elements": "lib",
    "sflu_components.simlib": "lib (simulation harness)",
    "tf_lib": "lib (plotting)",
    "models.matlib": "lib (models-local copy)",
    "models.components": "lib (models-local copy)",
    "models.components2": "lib (models-local copy)",
    "sflu.params": "params (ifo yaml -> params struct)",
    "sflu.models": "model",
    "sflu.models.budget": "model (shared noise budget)",
    "sflu.models.coupled_cavity": "model",
    "sflu.models.int_fd_sqz": "model",
    "sflu.models.filter_cavity": "model",
}

MODULES = [
    # --- library layer -------------------------------------------------
    {
        "path": "optics/test_lib.py",
        "title": "Matrix library",
        "section": "Library",
        "trust": "reviewed",
        "blurb": "The quadrature matrix algebra every model is built on: "
                 "rotations, mode-mismatch, and promotion to higher-order modes.",
        # pure numeric assertions on MatrixLib; nothing to plot
        "expect_figures": False,
    },
    # --- component / model layer ---------------------------------------
    {
        "path": "optics/test_simple_cavities.py",
        "title": "Simple cavities",
        "section": "Components",
        "trust": "reviewed",
    },
    {
        "path": "optics/test_radiation_pressure.py",
        "title": "Radiation pressure",
        "section": "Components",
        "trust": "reviewed",
    },
    {
        "path": "models/test_FP_basic.py",
        "title": "Fabry-Perot cavity (basic)",
        "section": "Models",
        "trust": "single-author",
    },
    {
        "path": "models/test_FP.py",
        "title": "Fabry-Perot cavity",
        "section": "Models",
        "trust": "single-author",
    },
    {
        "path": "models/test_FC.py",
        "title": "Filter cavity",
        "section": "Models",
        "trust": "single-author",
    },
    {
        "path": "models/test_simple_cav.py",
        "title": "Simple cavity",
        "section": "Models",
        "trust": "single-author",
    },
    {
        "path": "models/test_optical_spring.py",
        "title": "Optical spring",
        "section": "Models",
        "trust": "single-author",
    },
    {
        "path": "models/test_rp_mirror.py",
        "title": "Radiation-pressure mirror",
        "section": "Models",
        "trust": "single-author",
    },
    {
        "path": "pi/test_pi_gain.py",
        "title": "Parametric instability gain",
        "section": "Models",
        "trust": "single-author",
    },
    # --- internal squeezing ---------------------------------------------
    {
        "path": "fromgwinc/intsqz/test_CCwIntSqz.py",
        "title": "Coupled cavity with internal squeezing",
        "section": "Internal squeezing",
        "trust": "reference",
        "blurb": "The reference internal-squeezing model. This is the "
                 "best-known-good state of the code.",
    },
    {
        "path": "fromgwinc/intsqz/test_CCwIntFDSqz.py",
        "title": "Internal frequency-dependent squeezing",
        "section": "Internal squeezing",
        "trust": "single-author",
        "blurb": "Adds a detuned travelling-wave filter cavity inside the "
                 "signal-extraction cavity.",
    },
    {
        "path": "fromgwinc/intsqz/test_intFDsqz_sweeps.py",
        "title": "Internal FD squeezing: parameter sweeps",
        "section": "Internal squeezing",
        "trust": "single-author",
    },
]

# Example modules deliberately left undocumented, and why. Anything else
# without a page is reported by audit_coverage(), so it cannot pass unnoticed.
EXCLUDED = {
    "optics/test_DRFPMI.py":
        "skips at module level unless pygwinc provides gwinc.plant or "
        "gwinc.noise.quantum2; produces nothing to document until then",
    "models/test_optickle_RP.py":
        "pure Optickle/MATLAB cross-check, no SFLU model of its own",
    "models/test_simple_mirror.py":
        "test_sflu_simple_mirror currently fails; document once it passes",
    "fromgwinc/intsqz/test_FP.py":
        "smoke test for the parameter loader, no model and no figures",
    "fromgwinc/test_strain_single_arm.py":
        "single-arm strain example, deleted in the working tree but not yet "
        "committed; drop this entry along with the file",
}

# Examples needing Optickle/MATLAB or Finesse, which are optional dependencies.
PYTEST_K_FILTER = (
    "not optickle and not compare_optickle and not compare_sim and not cmp_FP"
)

SAVE_RE = re.compile(r"""tpath_join\(\s*['"]([^'"]+)['"]""")

# Figure stems like "A#cmp" and "A+cmp" are legal filenames but break markdown
# image URLs, so destination names are sanitised.
UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(name):
    return UNSAFE_RE.sub("_", name)


# ---------------------------------------------------------------------------
# running the examples
# ---------------------------------------------------------------------------

# The examples only write figures when pytest is given --plot, which comes from
# the wield.pytest plugin. Normally the plugin is autoloaded through its pytest11
# entry point, but that discovery is not reliable across environments: on a
# GitHub runner with pytest 9.1 and setuptools 84 it did not happen, and every
# page was built with no figures at all.
#
# Passing -p unconditionally is not an option either -- where autoload does work
# it raises "Plugin already registered under a different name". And probing with
# a separate, smaller pytest invocation is not reliable either: the answer turns
# out to depend on the rootdir the probe resolves, so the probe can disagree
# with the run it is supposed to predict.
#
# So: run the real command, and if pytest rejects --plot, run it again with the
# plugin named explicitly. No prediction, no divergence.
PLOT_PLUGIN = "wield.pytest.plugin"
PLOT_UNKNOWN = "unrecognized arguments: --plot"


def _pytest_cmd(paths, plugin_args=()):
    return [
        sys.executable, "-m", "pytest",
        "-s", "--plot",
        "-k", PYTEST_K_FILTER,
        "-p", "no:cacheprovider",
        *plugin_args,
        "--continue-on-collection-errors",
    ] + [str(ROOT / p) for p in paths]


def run_examples(paths):
    """Run the example modules so they deposit figures in tresults/."""
    # PYTHONHASHSEED is pinned for the same reason the regression baselines
    # pin it: SFLU's elimination order is set-iteration order, so unpinned
    # runs produce figures that wobble at the 1e-3 level between builds.
    env = dict(os.environ, PYTHONHASHSEED="0")

    for attempt, plugin_args in enumerate(([], ["-p", PLOT_PLUGIN])):
        cmd = _pytest_cmd(paths, plugin_args)
        if attempt:
            print(f"  retrying with {PLOT_PLUGIN} loaded explicitly")
        print("Running examples:\n  " + " ".join(cmd))
        result = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        print(output, end="" if output.endswith("\n") else "\n")
        if PLOT_UNKNOWN not in output:
            break
    else:
        print(f"  WARNING: pytest rejects --plot even with {PLOT_PLUGIN} "
              f"loaded, so the examples produced no figures. "
              f"Is wield-pytest installed?")

    if result.returncode != 0:
        print(f"  (pytest exited {result.returncode}; "
              f"building docs from whatever outputs exist)")
    return result.returncode


# ---------------------------------------------------------------------------
# literate chunking
# ---------------------------------------------------------------------------

def _dedent_block(lines):
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return lines
    pad = min(len(ln) - len(ln.lstrip()) for ln in body)
    return [ln[pad:] if ln.strip() else "" for ln in lines]


def _clean_prose(lines):
    """Turn a run of ``# comment`` lines into markdown prose."""
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        s = s.lstrip("#").strip()
        # Decorative rules like ###### or ---- carry no information.
        if not s or set(s) <= set("#-=*_"):
            continue
        # ...nor do the dashes wrapping "--- a section title ---".
        s = s.strip("-=*_ ").strip()
        if not s:
            continue
        out.append(s)
    return " ".join(out).strip()


def chunk_function(node, lines):
    """Split a function body into alternating prose / code chunks.

    A chunk boundary opens whenever a run of comment lines (or a bare string
    expression) is found between statements.  The result is a list of
    ``("prose"|"code", text)`` pairs in source order.
    """
    body = list(node.body)
    # Drop the docstring; it is rendered separately as the page intro.
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        return []

    chunks = []
    pending_code = []

    def flush_code():
        if not pending_code:
            return
        text = "\n".join(_dedent_block(pending_code)).strip("\n")
        if text.strip():
            chunks.append(("code", text))
        pending_code.clear()

    prev_end = node.body[0].lineno - 1 if node.body else node.lineno
    if body:
        prev_end = body[0].lineno - 1

    for i, stmt in enumerate(body):
        start = stmt.lineno - 1
        for dec in getattr(stmt, "decorator_list", []):
            start = min(start, dec.lineno - 1)

        # Comment lines sitting immediately above this statement.
        lead = []
        j = start - 1
        while j >= prev_end:
            s = lines[j].strip()
            if s.startswith("#"):
                lead.append(lines[j])
                j -= 1
            elif not s:
                j -= 1
            else:
                break
        lead.reverse()

        prose = _clean_prose(lead)
        # Only break the code block for a substantial comment; short trailing
        # notes like "# FIXME" read better inline with the code.
        if prose and len(prose) > 25:
            flush_code()
            chunks.append(("prose", prose))
        elif lead:
            pending_code.extend(lead)

        end = stmt.end_lineno
        pending_code.extend(lines[start:end])
        prev_end = end

    flush_code()
    return chunks


def parse_module(path):
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    module_doc = ast.get_docstring(tree) or ""

    functions = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not (node.name.startswith("test_") or node.name.startswith("plot_")):
            continue
        functions.append({
            "name": node.name,
            "doc": ast.get_docstring(node) or "",
            "chunks": chunk_function(node, lines),
        })
    return module_doc, functions, tree


def repo_imports(tree):
    """Imports resolving to modules inside this repository, with their layer."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:  # relative import inside fromgwinc.intsqz
                mod = f"fromgwinc.intsqz.{mod}" if mod else "fromgwinc.intsqz"
            names = ", ".join(a.name for a in node.names)
            found.append((mod, names))
        elif isinstance(node, ast.Import):
            for a in node.names:
                found.append((a.name, ""))

    rows = []
    seen = set()
    for mod, names in found:
        layer = LAYERS.get(mod)
        if layer is None:
            # `import components as cmp` style, resolved relative to models/
            layer = LAYERS.get(f"models.{mod}")
        if layer is None or mod in seen:
            continue
        seen.add(mod)
        rows.append((mod, layer, names))
    return sorted(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def pdf_to_png(pdf, png, dpi=140):
    try:
        r = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
             str(pdf), str(png.with_suffix(""))],
            capture_output=True,
        )
        return r.returncode == 0 and png.exists()
    except FileNotFoundError:
        return False


def _figures_in(tdir):
    figs = {}
    for png in sorted(tdir.glob("*.png")):
        figs[png.stem] = png
    for pdf in sorted(tdir.glob("*.pdf")):
        if pdf.stem in figs:
            continue
        png = pdf.with_suffix(".png")
        if png.exists() or pdf_to_png(pdf, png):
            figs[pdf.stem] = png
    return figs


def collect_figures(module_path, test_name):
    """Map stem -> png path for every figure an example produced.

    A parametrized example writes to ``tresults/<name>[<param>]`` rather than
    to ``tresults/<name>``, so match those too and keep the parameter id in the
    key. Without this every figure from a parametrized example is silently
    dropped from the docs.
    """
    troot = module_path.parent / "tresults"
    if not troot.is_dir():
        return {}

    figs = {}
    exact = troot / test_name
    if exact.is_dir():
        figs.update(_figures_in(exact))

    for tdir in sorted(troot.glob(f"{test_name}[[]*[]]")):
        if not tdir.is_dir():
            continue
        param = tdir.name[len(test_name):].strip("[]")
        for stem, png in _figures_in(tdir).items():
            figs[f"{stem} [{param}]" if param else stem] = png
    return figs


# ---------------------------------------------------------------------------
# page rendering
# ---------------------------------------------------------------------------

def render_page(info):
    path = ROOT / info["path"]
    module_doc, functions, tree = parse_module(path)

    out = [f"# {info['title']}\n"]

    label, note = TRUST[info["trust"]]
    kind = {"reviewed": "note", "reference": "info",
            "single-author": "warning"}[info["trust"]]
    out.append(f'!!! {kind} "Review status: {label}"\n')
    out.append(f"    {note}\n")

    if info.get("blurb"):
        out.append(f"{info['blurb']}\n")
    if module_doc:
        out.append(f"{module_doc}\n")

    out.append(f"**Source:** `{info['path']}`\n")

    rows = repo_imports(tree)
    if rows:
        out.append("### Depends on\n")
        out.append("| module | layer | imported |")
        out.append("|---|---|---|")
        for mod, layer, names in rows:
            out.append(f"| `{mod}` | {layer} | {names or '—'} |")
        out.append("")

    # Image links are relative to the page, which may sit one or two
    # directories deep under docs/docs/.
    img_prefix = "../" * len(Path(info["path"]).parent.parts) + "img"

    copies = []
    figure_count = 0
    for func in functions:
        out.append(f"## `{func['name']}`\n")
        if func["doc"]:
            out.append(f"{func['doc']}\n")

        figs = collect_figures(path, func["name"])
        figure_count += len(figs)
        used = set()

        def emit(key, _figs=figs, _func=func):
            # The key carries the parameter id, so it -- not the source
            # filename -- names the copy: two parameter runs of one example
            # write the same filename in different directories.
            png = _figs[key]
            dest = safe_name(f"{path.stem}__{_func['name']}__{key}{png.suffix}")
            copies.append((png, dest))
            out.append(f"![{key}]({img_prefix}/{dest})\n")

        for kind_, text in func["chunks"]:
            if kind_ == "prose":
                out.append(f"{text}\n")
                continue
            out.append(f"```python\n{text}\n```\n")
            # Embed any figure this chunk saved, right here. One saved stem can
            # correspond to several keys, one per parameter id.
            for saved in SAVE_RE.findall(text):
                saved = Path(saved).stem
                for key in sorted(figs):
                    if key != saved and not key.startswith(f"{saved} ["):
                        continue
                    if key in used:
                        continue
                    used.add(key)
                    emit(key)

        leftover = [s for s in figs if s not in used]
        if leftover:
            out.append("### Output\n")
            for key in sorted(leftover):
                emit(key)

    return "\n".join(out), copies, figure_count


def render_index(built):
    out = [
        "# SFLU Optomechanics",
        "",
        "Signal-flow (SFLU) models of optomechanical interferometers, with "
        "internal squeezing.",
        "",
        "Every page below is generated directly from a runnable example in the "
        "repository: the prose is the example's own comments, the figures are "
        "the ones it produced on this build. Nothing here is written twice, so "
        "the documentation cannot drift from the code.",
        "",
        "## How the code is layered",
        "",
        "```",
        "params (*.yaml)  ->  ifo Struct",
        "        |",
        "        v",
        "lib      sflu_components/{lib,edges,elements}.py   quadrature algebra,",
        "                                                   edge + graph components",
        "        |",
        "        v",
        "model    topology (SFLU graph)  ->  plant (edges -> transfer functions)",
        "                                ->  budget (transfer functions -> PSD)",
        "        |",
        "        v",
        "example  test_*.py               runs a model, makes figures",
        "```",
        "",
        "See `REFACTOR_PLAN.md` and `docs/DEPENDENCIES.md` in the repository "
        "for the current state of that layering and the plan to make it "
        "explicit.",
        "",
        "## Review status",
        "",
        "Pages are labelled with how much review the underlying code has had:",
        "",
        "| label | meaning |",
        "|---|---|",
    ]
    for _key, (label, note) in TRUST.items():
        out.append(f"| **{label}** | {note} |")
    out.append("")
    out.append("## Examples")
    out.append("")

    sections = {}
    for info in built:
        sections.setdefault(info["section"], []).append(info)
    for section, entries in sections.items():
        out.append(f"### {section}")
        out.append("")
        for info in entries:
            p = Path(info["path"])
            out.append(f"- [{info['title']}]({p.parent}/{p.stem}.md)")
        out.append("")
    return "\n".join(out)


def render_mkdocs(built):
    sections = {}
    for info in built:
        sections.setdefault(info["section"], []).append(info)
    nav = ["  - Home: index.md"]
    for section, entries in sections.items():
        nav.append(f'  - "{section}":')
        for info in entries:
            p = Path(info["path"])
            # Quoted: titles may contain a colon, which is YAML-significant.
            title = info["title"].replace('"', "'")
            nav.append(f'    - "{title}": {p.parent}/{p.stem}.md')
    nav_block = "\n".join(nav)
    return f"""# Generated by docs/generate_docs.py -- do not edit by hand.
site_name: SFLU Optomechanics
site_description: Literate documentation generated from the example suite
docs_dir: docs

theme:
  name: material
  palette:
    scheme: default
    primary: indigo
  features:
    - navigation.sections
    - content.code.copy

nav:
{nav_block}

markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
"""


def audit_coverage():
    """Example modules in the repo that no page is generated from.

    Not automatically an error -- some are excluded on purpose, see EXCLUDED --
    but an unexplained one means a new example silently has no documentation.
    """
    listed = {m["path"] for m in MODULES}
    skip_parts = {".git", "attic", "tools", "__pycache__", "docs", "deps"}
    undocumented = []
    for path in sorted(ROOT.rglob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if skip_parts & set(Path(rel).parts):
            continue
        if rel not in listed and rel not in EXCLUDED:
            undocumented.append(rel)
    return undocumented


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-tests", action="store_true",
                    help="reuse existing tresults/ output instead of re-running")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if an example is missing, fails, or "
                         "produces no figures when it is expected to")
    args = ap.parse_args()

    problems = []

    present = [m for m in MODULES if (ROOT / m["path"]).exists()]
    for m in MODULES:
        if m not in present:
            problems.append(f"listed module not found: {m['path']}")
            print(f"  warning: {m['path']} not found, skipping")

    if args.skip_tests:
        print("Skipping example run; using existing tresults/")
    else:
        rc = run_examples([m["path"] for m in present])
        if rc != 0:
            problems.append(f"example run exited {rc}")

    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    built, all_copies = [], []
    for info in present:
        page, copies, n_figs = render_page(info)
        p = Path(info["path"])
        md = SITE_SRC / p.parent / f"{p.stem}.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(page)
        built.append(info)
        all_copies.extend(copies)
        print(f"  page: {md.relative_to(SITE_SRC)}  ({len(copies)} figures)")
        if info.get("expect_figures", True) and n_figs == 0:
            problems.append(
                f"no figures from {info['path']} -- if that is correct, set "
                f'"expect_figures": False on its MODULES entry'
            )

    for src, dest in all_copies:
        shutil.copy2(src, IMG_DIR / dest)

    (SITE_SRC / "index.md").write_text(render_index(built))
    (DOCS / "mkdocs.yml").write_text(render_mkdocs(built))
    print(f"\n{len(built)} pages, {len(all_copies)} figures")

    undocumented = audit_coverage()
    if undocumented:
        print("\nExample modules with no documentation page:")
        for rel in undocumented:
            print(f"  {rel}")
        print("Add each to MODULES, or to EXCLUDED with a reason.")
        problems.append(f"{len(undocumented)} undocumented example module(s)")

    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"  - {problem}")
        if args.strict:
            print("\nFailing because --strict was given.")
            return 1

    print("\nPreview:  python -m mkdocs serve -f docs/mkdocs.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
