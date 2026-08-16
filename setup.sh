#!/usr/bin/env bash
# Create the conda environment this repo runs in.
#
# Idempotent: safe to re-run. Clones dev dependencies into deps/ and installs
# them editable, so `git pull` in deps/<name> is enough to update one.
#
# Adapted from the setup.sh on the refactor/intsqz branch, with two changes:
# wield-pytest is now installed (it provides the --plot option the docs
# generator passes, via its pytest11 entry point), and gwinc comes from the
# package index rather than a fork -- see docs/GWINC_DEPENDENCY.md for why the
# fork dependency was dropped.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-wield}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${REPO_DIR}/deps"

# local_dir  git_url  branch
DEV_DEPS=(
  "wield-control     ssh://git@git.mccullerlab.com:2224/Jeffrey.Wack/wield-control.git  fix/mimo-ss-keyword"
  "wield-bunch       ssh://git@git.mccullerlab.com:2224/wield/wield-bunch.git           main"
  "wield-utilities   ssh://git@git.mccullerlab.com:2224/wield/wield-utilities.git       main"
  "wield-declarative ssh://git@git.mccullerlab.com:2224/wield/wield-declarative.git     main"
  "wield-pytest      ssh://git@git.mccullerlab.com:2224/wield/wield-pytest.git          main"
)

# LaTeX is needed because the examples render labels with text.usetex=True;
# poppler-utils supplies pdftoppm, which the docs generator uses to turn
# figure PDFs into PNGs.
APT_PACKAGES=(
  texlive-latex-base
  texlive-fonts-recommended
  texlive-latex-extra
  cm-super
  dvipng
  poppler-utils
)

PIP_PACKAGES=(
  numpy scipy matplotlib pyyaml networkx h5py ipython
  control
  "gwinc==0.6.2"
  pytest pytest-watcher
  mkdocs mkdocs-material
)

log() { echo "== $*"; }

check_conda() {
    if ! command -v conda &>/dev/null; then
        echo "ERROR: conda not found. Install miniforge or miniconda first." >&2
        exit 1
    fi
}

install_apt_deps() {
    log "system packages"
    if ! command -v apt-get &>/dev/null; then
        echo "   no apt-get; ensure these are present by other means:"
        printf '     %s\n' "${APT_PACKAGES[@]}"
        return
    fi
    local missing=()
    for pkg in "${APT_PACKAGES[@]}"; do
        dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
    done
    if [ ${#missing[@]} -eq 0 ]; then
        echo "   already installed"
    else
        echo "   installing: ${missing[*]}"
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update && apt-get install -y "${missing[@]}"
        else
            sudo apt-get update && sudo apt-get install -y "${missing[@]}"
        fi
    fi
}

create_conda_env() {
    log "conda environment: ${CONDA_ENV}"
    if conda env list | grep -qE "^${CONDA_ENV}\s"; then
        echo "   already exists"
    else
        conda create -y -n "${CONDA_ENV}" python="${PYTHON_VERSION}"
    fi
    # slycot is a compiled dependency of python-control; conda-forge has wheels
    conda install -y -n "${CONDA_ENV}" -c conda-forge slycot
}

install_pip_packages() {
    log "pip packages"
    conda run -n "${CONDA_ENV}" pip install "${PIP_PACKAGES[@]}"
}

install_dev_deps() {
    log "dev dependencies in ${DEPS_DIR}/"
    mkdir -p "${DEPS_DIR}"
    for dep in "${DEV_DEPS[@]}"; do
        read -r dir url branch <<< "${dep}"
        local target="${DEPS_DIR}/${dir}"
        if [ -d "${target}" ]; then
            echo "   [skip]  ${dir} (already cloned)"
        else
            echo "   [clone] ${dir} @ ${branch}"
            git clone --depth 1 -b "${branch}" "${url}" "${target}"
        fi
        conda run -n "${CONDA_ENV}" pip install -e "${target}"
    done
}

verify() {
    log "verifying"
    conda run -n "${CONDA_ENV}" python - <<'PY'
import importlib
missing = []
for mod in ("numpy", "scipy", "matplotlib", "networkx", "gwinc",
            "wield.control", "wield.bunch", "wield.utilities", "wield.pytest"):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("FAILED to import:\n  " + "\n  ".join(missing))
print("   all required modules import")
PY
    command -v pdftoppm >/dev/null \
        && echo "   pdftoppm present (figure PDF -> PNG)" \
        || echo "   WARNING: pdftoppm missing; docs figures from PDFs will be skipped"
}

check_conda
install_apt_deps
create_conda_env
install_pip_packages
install_dev_deps
verify

cat <<EOF

Setup complete.

  conda activate ${CONDA_ENV}
  make test     # run the suite
  make guard    # check model outputs against the stored baselines
  make docs     # run the examples and rebuild the documentation
EOF
