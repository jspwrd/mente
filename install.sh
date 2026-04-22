#!/usr/bin/env bash
# mente installer — picks the best available Python package manager and
# drops you at a working `mente` on your PATH.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/jspwrd/mente/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/jspwrd/mente/main/install.sh | bash -s -- --with=llm
#   ./install.sh --with=llm,embeddings-local
#
# Flags:
#   --with=<extras>   Comma-separated optional-deps to include
#                     (llm, embeddings, embeddings-local, dev, docs).
#                     Repeat the flag or comma-join — both work.
#   --pre             Install the latest pre-release (not recommended).
#   --version=X.Y.Z   Pin a specific version.
#   --no-uv           Skip the uv-tool path even if uv is available.
#   --dry-run         Print what would run and exit.
#   -h, --help        This help.
#
# The installer tries, in order:
#   1. `uv tool install` — preferred; ships a dedicated venv, no conflicts
#   2. `pipx install`    — same shape, different tool
#   3. `pip install --user` — last resort; puts `mente` in ~/.local/bin
#
# Python 3.11+ required. The installer will refuse to proceed on older versions.

set -euo pipefail

PACKAGE="mente"
EXTRAS=""
VERSION=""
PRE_RELEASE="false"
USE_UV="true"
DRY_RUN="false"

log()  { printf '\033[1;34m[mente]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[mente]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[mente]\033[0m %s\n' "$*" >&2; exit 1; }

show_help() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with=*)       EXTRAS="${EXTRAS:+$EXTRAS,}${1#--with=}" ;;
    --with)         shift; EXTRAS="${EXTRAS:+$EXTRAS,}$1" ;;
    --version=*)    VERSION="${1#--version=}" ;;
    --version)      shift; VERSION="$1" ;;
    --pre)          PRE_RELEASE="true" ;;
    --no-uv)        USE_UV="false" ;;
    --dry-run)      DRY_RUN="true" ;;
    -h|--help)      show_help ;;
    *)              die "unknown flag: $1 (try --help)" ;;
  esac
  shift
done

spec="$PACKAGE"
if [ -n "$EXTRAS" ]; then
  extras_normalized="$(printf '%s' "$EXTRAS" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)"
  spec="${spec}[${extras_normalized}]"
fi
if [ -n "$VERSION" ]; then
  spec="${spec}==${VERSION}"
fi

find_python() {
  for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
  done
  return 1
}

PY="$(find_python || true)"
[ -n "$PY" ] || die "no python3 interpreter found. install Python 3.11+ and retry."

ver="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
maj="${ver%.*}"; min="${ver#*.}"
if [ "$maj" -lt 3 ] || { [ "$maj" -eq 3 ] && [ "$min" -lt 11 ]; }; then
  die "Python 3.11+ required; found $ver ($PY)."
fi
log "using Python $ver at $(command -v "$PY")"

run() {
  if [ "$DRY_RUN" = "true" ]; then
    printf '\033[2m  would run:\033[0m %s\n' "$*"
    return 0
  fi
  "$@"
}

install_via_uv() {
  log "installing with uv tool install"
  local args=(tool install "$spec")
  [ "$PRE_RELEASE" = "true" ] && args+=(--prerelease allow)
  run uv "${args[@]}"
}

install_via_pipx() {
  log "installing with pipx"
  local args=(install "$spec")
  [ "$PRE_RELEASE" = "true" ] && args+=(--pip-args=--pre)
  run pipx "${args[@]}"
}

install_via_pip() {
  log "installing with pip --user (fallback)"
  local args=(-m pip install --user --upgrade "$spec")
  [ "$PRE_RELEASE" = "true" ] && args+=(--pre)
  run "$PY" "${args[@]}"
  if ! printf '%s' ":$PATH:" | grep -q ":$HOME/.local/bin:"; then
    warn "~/.local/bin is not on your PATH. Add:"
    warn '    export PATH="$HOME/.local/bin:$PATH"'
  fi
}

if [ "$USE_UV" = "true" ] && command -v uv >/dev/null 2>&1; then
  install_via_uv
elif command -v pipx >/dev/null 2>&1; then
  install_via_pipx
else
  install_via_pip
fi

if [ "$DRY_RUN" = "true" ]; then
  log "dry run complete — no changes made."
  exit 0
fi

if command -v mente >/dev/null 2>&1; then
  log "installed: $(command -v mente)"
  log "try: mente   (or: mente --help)"
else
  warn "mente was installed but isn't on PATH yet."
  warn "open a new shell, or source your shell rc, then try: mente --help"
  exit 1
fi
