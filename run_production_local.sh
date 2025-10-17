#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="${SCRIPT_DIR}/venv"
AUTO_CONFIRM=1
declare -a extra_args
extra_args=()

print_header() {
    cat <<'EOF'
==========================================
AI Video Editor - Production Local Run
==========================================
EOF
}

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [-y|--yes] <video_file> [ai-video-editor-local options...]

Examples:
  ${SCRIPT_NAME} my_video.mp4
  ${SCRIPT_NAME} -y my_video.mp4 --parts 8 --skip-editing
  ${SCRIPT_NAME} my_video.mp4 --device cuda --torch-dtype bfloat16

Flags:
  -y, --yes        Skip the first-run confirmation prompt.
  -h, --help       Show this message.

All additional arguments are forwarded to 'ai-video-editor-local'.
EOF
}

die() {
    echo "ERROR: $1" >&2
    exit 1
}

confirm_notice() { :; }

activate_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        die "Virtual environment not found at ${VENV_DIR}. Run 'python3 -m venv venv && source venv/bin/activate && pip install -e .' first."
    fi
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
}

ensure_file() {
    local path="$1"
    [[ -f "${path}" ]] || die "Video file not found: ${path}"
}

resolve_output_dir() {
    local video_path="$1"
    local override=""

    for ((idx=0; idx<${#extra_args[@]}; idx++)); do
        local value="${extra_args[$idx]}"
        if [[ "${value}" == "--output-dir" || "${value}" == "-o" ]]; then
            local next_idx=$((idx + 1))
            if (( next_idx < ${#extra_args[@]} )); then
                override="${extra_args[$next_idx]}"
            fi
        fi
    done

    if [[ -n "${override}" ]]; then
        echo "${override}"
        return
    fi

    local base
    base=$(basename "${video_path}")
    base="${base%.*}"
    override="${base}_ai_edited"
    if (( ${#extra_args[@]} > 0 )); then
        extra_args=("--output-dir" "${override}" "${extra_args[@]}")
    else
        extra_args=("--output-dir" "${override}")
    fi
    echo "${override}"
}

ensure_option() {
    local flag="$1"
    local value="$2"
    local found=0
    for ((idx=0; idx<${#extra_args[@]}; idx++)); do
        if [[ "${extra_args[$idx]}" == "${flag}" ]]; then
            found=1
            break
        fi
    done
    if (( found == 0 )); then
        extra_args+=("${flag}" "${value}")
    fi
}

summarize_success() {
    local output_dir="$1"
    cat <<EOF

==========================================
✅ Processing Complete!
==========================================

Output location: ${output_dir}/

Generated files:
  📁 chunks/        - Video segments
  📁 analysis/      - AI analysis reports
  📁 plan/          - Editing plan (JSON + Markdown)
  📁 new/           - Edited segments
  🎬 final_video.mp4 - Your finished video!

To view the final video:
  open "${output_dir}/final_video.mp4"
EOF
}

summarize_failure() {
    local exit_code="$1"
    cat <<EOF

==========================================
❌ Processing Failed
==========================================
Exit code: ${exit_code}

Common issues:
  - Model download interrupted (press Ctrl+C to cancel, then retry)
  - Out of memory (try --torch-dtype float16 or smaller models)
  - Disk space (need 25GB+ for models)
EOF
}

main() {
    print_header

    extra_args=()
    local video_file=""

    while (( $# )); do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -y|--yes)
                AUTO_CONFIRM=1
                ;;
            --)
                shift
                extra_args+=("$@")
                break
                ;;
            -*)
                extra_args+=("$1")
                ;;
            *)
                if [[ -z "${video_file}" ]]; then
                    video_file="$1"
                else
                    extra_args+=("$1")
                fi
                ;;
        esac
        shift || true
    done

    if [[ -z "${video_file}" ]]; then
        usage
        exit 1
    fi

    ensure_file "${video_file}"

    confirm_notice
    activate_venv

    local output_dir
    output_dir=$(resolve_output_dir "${video_file}")

    ensure_option "--device" "auto"
    ensure_option "--torch-dtype" "auto"
    ensure_option "--log-level" "INFO"

    echo "Video file: ${video_file}"
    echo ""
    echo "=========================================="
    echo "Starting Production Processing"
    echo "=========================================="
    echo "Output directory: ${output_dir}"
    if (( ${#extra_args[@]} > 0 )); then
        echo "Additional options: ${extra_args[*]}"
    else
        echo "Additional options: (none)"
    fi
    echo ""

    if ! command -v ai-video-editor-local >/dev/null 2>&1; then
        die "'ai-video-editor-local' not found. Ensure the project is installed into the virtual environment."
    fi

    ai-video-editor-local "${video_file}" "${extra_args[@]}"
    local exit_code=$?

    if [[ "${exit_code}" -eq 0 ]]; then
        summarize_success "${output_dir}"
    else
        summarize_failure "${exit_code}"
    fi

    exit "${exit_code}"
}

main "$@"
