#!/bin/bash

set -e


REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
SCRIPTS_DIR="${REPO_ROOT_DIR}/utility"

source "${SCRIPTS_DIR}/logging.sh"


teardown_nethack_conda_env() {
  log_info "Removing nethack conda environment..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env remove -n nethack
  log_success "Removed nethack conda environment!"
}


teardown_procgen_env
