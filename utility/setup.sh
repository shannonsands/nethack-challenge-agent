#!/bin/bash

set -e

MINICONDA_DOWNLOAD_PATH="/tmp/miniconda.sh"
REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
SCRIPTS_DIR="${REPO_ROOT_DIR}/utility"
NETHACK_PYPI_PACKAGE_NAME="git+https://github.com/facebookresearch/nle.git"


source "${SCRIPTS_DIR}/logging.sh"


_download_miniconda() {
  log_info "Downloading Miniconda..."
  miniconda_base_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-"
  case `uname` in
  "Linux")
    miniconda_file="Linux-x86_64.sh"
    ;;
  "Darwin")
    miniconda_file="MacOSX-x86_64.sh"
    ;;
  *)
    log_error "Sorry, we don't support this platform :("
    ;;
  esac
  wget -O "${MINICONDA_DOWNLOAD_PATH}" "${miniconda_base_url}${miniconda_file}"
  log_success "Downloaded Miniconda!"
}


download_miniconda() {
  if [ -f "${MINICONDA_DOWNLOAD_PATH}" ]; then
    return
  fi
  _download_miniconda
}


_install_miniconda() {
  log_info "Installing Miniconda..."
  bash ${MINICONDA_DOWNLOAD_PATH} -b -p ${HOME}/miniconda3
  . ${HOME}/miniconda3/etc/profile.d/conda.sh
  conda init
}


install_miniconda() {
  if which conda 2> /dev/null 1>&2; then
    log_info "Found an existing conda installation!"
    return
  fi
  download_miniconda
  _install_miniconda
}


_setup_nethack_conda_env() {
  log_info "Creating nethack conda environment..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -n nethack -y
  conda activate nethack
  conda install python cmake -y
  log_success "Created nethack conda environment!"
}


_install_nethack() {
  log_info "Installing NetHack gym environment..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate nethack
  conda install -y numpy scikit-learn
  pip install -U "${NETHACK_PYPI_PACKAGE_NAME}" aicrowd-gym
  pip install -r "${REPO_ROOT_DIR}/requirements.txt"
  log_success "Installed NetHack gym environment!"
}


setup_nethack_conda_env() {
  if conda activate nethack 2> /dev/null 1>&2; then
    log_info "Re-using existing nethack conda environment..."
  else
    _setup_nethack_conda_env
  fi
  _install_nethack
}


install_miniconda
setup_nethack_conda_env

log_info "You can start using NetHack conda environment by running"
log_normal "\`conda activate nethack\`"

log_info "**Note:** Please restart your terminal if you did not have conda previously installed."
