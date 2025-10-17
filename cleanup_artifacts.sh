#!/usr/bin/env bash

set -euo pipefail

DEFAULT_OUTPUT_DIR="ai_video_editor_output"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--output-dir DIR]

Removes generated artifacts for the AI Video Editor, including:
  - chunks/, new/, edited/, quality/ directories under the output root
  - final rendered videos (*.mp4) at the root level
  - ffmpeg manifests (render_manifest.json)
  - intermediate analysis attempts (chunk_*_attempt*_raw.json)

Options:
  --output-dir DIR   Target directory (default: ${DEFAULT_OUTPUT_DIR})
  -h, --help         Show this message
EOF
}

OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"

while (( $# )); do
    case "$1" in
        --output-dir)
            shift
            OUTPUT_DIR="${1:-}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift || true
done

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: --output-dir requires a value" >&2
    exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory not found (nothing to clean): ${OUTPUT_DIR}"
    exit 0
fi

echo "Cleaning artifacts under ${OUTPUT_DIR}"

for dir in chunks new edited quality; do
    target="${OUTPUT_DIR}/${dir}"
    if [[ -d "${target}" ]]; then
        rm -rf "${target}"
        echo "Removed directory: ${target}"
    fi
done

find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' -print -delete

analysis_dir="${OUTPUT_DIR}/analysis"
if [[ -d "${analysis_dir}" ]]; then
    find "${analysis_dir}" -type f -name 'chunk_*_attempt*_raw.json' -print -delete
fi

find "${OUTPUT_DIR}" -maxdepth 2 -type f -name 'render_manifest.json' -print -delete

echo "Cleanup complete."
