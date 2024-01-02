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

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null
}

# Check required dependencies
check_dependencies() {
    if ! command_exists docker; then
        cat <<EOF

Docker is not installed. Install it using:

curl -fsSL https://get.docker.com -o get-docker.sh && \\
    sudo sh get-docker.sh

EOF
        exit 1
    fi

    if ! command_exists sam; then
        cat <<EOF

AWS SAM CLI is not installed. Install it with:

wget https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip && \\
    unzip aws-sam-cli-linux-x86_64.zip -d sam-installation && \\
    sudo ./sam-installation/install && \\
    rm -rf aws-sam-cli-linux-x86_64.zip sam-installation

EOF
        exit 1
    fi

    if ! command_exists pip; then
        cat <<EOF

Pip is not installed. Install it with:

Linux:

sudo apt install python3-pip

MacOS:

brew install python3-pip

EOF
        exit 1
    fi
}

# Main function of the script
main() {
    local FUNCTION="cerebro"

    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -h|--help)
                help_function
                exit 0
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

    # Check for required dependencies
    check_dependencies

    # Install virtual environment 
    echo "Activating environment..."
    python3 -m venv venv && \
        source venv/bin/activate && \
        pip install -q --upgrade pip && \
        pip install -q -r requirements.txt

    # Deploy updates to AWS
    echo "Updating endpoint..."
    black -q cerebro/ && \
        sam validate && \
        sam build && \
        PYTHONPATH="$(pwd)"/cerebro pytest -q tests/unit && \
        deactivate && \
        sam sync --stack-name sam-app --watch
}

main "$@"
