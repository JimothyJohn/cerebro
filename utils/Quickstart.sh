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
        echo "Docker is not installed. Install it using:"
        echo "curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh"
        exit 1
    fi

    if ! command_exists aws; then
        echo "AWS CLI is not installed. Install it from:"
        echo "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        exit 1
    fi

    if ! command_exists sam; then
        echo "AWS SAM CLI is not installed. Install it from:"
        echo "https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
        exit 1
    fi

    if ! command_exists python3 || ! command_exists pip || ! command_exists pipenv; then
        echo "One or more Python related dependencies (Python3, pip, pipenv) are not installed."
        echo "Please install them for your platform."
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

    # Deploy updates to AWS
    pushd lambda/ > /dev/null && \
        pipenv run black cerebro/ && \
        sam validate && \
        sam build && \
        pipenv run python -m pytest tests/unit && \
        sam sync --stack-name sam-app --watch && \
        popd > /dev/null
}

main "$@"
