#!/usr/bin/env bash
# generate_sbom.sh — emit a CycloneDX SBOM for openrabbit (PRD §12 self-supply-chain).
#
# Uses a LAZILY-INSTALLED CycloneDX tool via `uvx` (the uv tool runner) so the
# generator is NEVER a declared runtime/dev dependency — keeping the supply chain
# (and install time) lean. `uvx cyclonedx-py` is fetched on demand into an
# ephemeral, isolated environment and discarded after.
#
# Output: sbom.json (CycloneDX 1.6 JSON) at the repo root. Commit it (and bump it
# whenever dependencies change) so consumers can vet openrabbit's dependency tree.
#
# Usage:
#   ./scripts/generate_sbom.sh            # -> ./sbom.json
#   ./scripts/generate_sbom.sh out.json   # -> ./out.json
#
# Requirements: `uv` (which provides `uvx`). No network access in CI is needed at
# review time — this is a maintenance/release script, not part of the review path.
set -euo pipefail

# Resolve the repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT="${1:-${REPO_ROOT}/sbom.json}"

if ! command -v uvx >/dev/null 2>&1; then
  echo "error: 'uvx' not found. Install uv (https://docs.astral.sh/uv/) first." >&2
  exit 1
fi

echo "openrabbit: generating CycloneDX SBOM -> ${OUT}" >&2

# `cyclonedx-py environment` introspects an installed environment. We run it
# against the project's locked environment. The tool is fetched lazily by uvx.
#   --output-format JSON   -> CycloneDX JSON
#   --output-file <path>   -> write sbom.json (instead of stdout)
#   --pyproject <file>     -> include the root component metadata (name/license)
#   --output-reproducible  -> drop time/random values so the committed SBOM is
#                             diff-stable across regenerations
# We resolve the project's interpreter via `uv run` so the SBOM reflects the
# pinned (uv.lock) dependency set rather than whatever Python is on PATH.
PYTHON_BIN="$(cd "${REPO_ROOT}" && uv run python -c 'import sys; print(sys.executable)')"

uvx --from cyclonedx-bom cyclonedx-py environment \
  --output-format JSON \
  --output-file "${OUT}" \
  --pyproject "${REPO_ROOT}/pyproject.toml" \
  --output-reproducible \
  "${PYTHON_BIN}"

echo "openrabbit: wrote SBOM to ${OUT}" >&2
