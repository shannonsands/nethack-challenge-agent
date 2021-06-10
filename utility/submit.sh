#!/bin/bash

set -e

REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
SCRIPTS_DIR="${REPO_ROOT_DIR}/utility"

source "${SCRIPTS_DIR}/logging.sh"


print_usage() {
cat << USAGE
Usage: ./utility/submit.sh "impala-ppo-v0.1"
USAGE
}


bad_remote_message() {
  log_normal "AIcrowd remote not found"
  log_error "Please run \`git remote add aicrowd git@gitlab.aicrowd.com:<username>/<repo>.git\` and rerun this command"
  exit 1
}

check_remote() {
  log_info Checking git remote settings...

  bad_remotes=(
    git@gitlab.aicrowd.com:nethack/neurips-2021-the-nethack-challenge.git
    http://gitlab.aicrowd.com/nethack/neurips-2021-the-nethack-challenge.git
  )
  for bad_remote in $bad_remotes; do
    if git remote -v | grep "$bad_remote" > /dev/null; then
      bad_remote_message
    fi
  done

  if ! git remote -v | grep "gitlab.aicrowd.com"; then
    bad_remote_message
  fi
}


setup_lfs() {
  git lfs install
  HTTPS_REMOTE=$(git remote -v | grep gitlab.aicrowd.com | head -1 | awk '{print $2}' | sed 's|git@gitlab.aicrowd.com:|https://gitlab.aicrowd.com|g')
  git config lfs.$HTTPS_REMOTE/info/lfs.locksverify false
  find . -type f -size +5M -exec git lfs track {} &> /dev/null \;
  git add .gitattributes
}


setup_commits() {
  REMOTE=$(git remote -v | grep gitlab.aicrowd.com | head -1 | awk '{print $1}')
  TAG=$(echo "$@" | sed 's/ /-/g')
  git add --all
  git commit -m "Changes for submission-$TAG" || true
  git tag -am "submission-$TAG" "submission-$TAG" || (log_error "There is another submission with the same description. Please give a different description." && exit 1)
  git push -f $REMOTE master
  git push -f $REMOTE "submission-$TAG"
}


submit() {
  check_remote
  setup_lfs
  setup_commits
}
  


if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

submit "$@"
