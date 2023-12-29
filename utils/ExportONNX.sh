#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -e

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

# Function to display help
help_function() {
    cat <<EOF
Usage: $(basename "$0") [-h] [--lib-dir LIB_DIR]

Options:
  -h, --help       Display this help message and exit
  --lib-dir        Optional path to the lib/ directory
  -f, --function   Name of the lambda function
EOF
}

# Main function of the script
main() {
    local LIB_DIR=""
    local FUNCTION="cerebro"

    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -h|--help)
                help_function
                exit 0
                ;;
            --lib-dir)
                LIB_DIR="$2"
                shift 2
                ;;
            -f|--function)
                FUNCTION="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter passed: $1"
                help_function
                exit 1
                ;;
        esac
    done

    # Deploy updates to AWS
    docker run --rm -it \
        -v $(pwd):/workspace \
        -w /workspace \
        ultralytics/ultralytics:latest-arm64 \
        yolo detect export model=yolov8n.pt format=onnx && \
        rm yolov8n.pt
}

main "$@"
