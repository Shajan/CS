#!/bin/bash

# Rebases all local branches to latest master
set -e

WS=$(git rev-parse --show-toplevel)
MASTER_BRANCH=${1:-master}
TMP_BRANCH=${2:-tmp.rebasing}

# Save current folder and branch
pushd "${WS}"
original_branch=$(git symbolic-ref --short HEAD)

if git rev-parse --verify "${TMP_BRANCH}" >/dev/null 2>&1; then
  echo "Aborting: Temporary branch '${TMP_BRANCH}' exists."
  exit 1
  popd
fi

# Check if there are uncommitted changes
if [[ $(git status --porcelain) ]]
then
  echo "Aborging: There are uncommitted changes in the repository."
  exit 1
  popd
fi

# Get latest master
git checkout "${MASTER_BRANCH}"

echo "[Updating "${MASTER_BRANCH}"] ----------"
git pull

for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/)
do
  if [ "${branch}" != "${MASTER_BRANCH}" ]
  then
    # Create temp branch to check for rebase conflict
    git checkout -b "${TMP_BRANCH}" "${branch}"

    if git rebase "${MASTER_BRANCH}" 2>&1 | grep -q "CONFLICT"
    then
      echo "[Skipping ${branch}, confict] ----------"
      git checkout "${branch}"
    else
      echo "[Rebasing ${branch}] ----------"
      git checkout "${branch}"
      git rebase "${MASTER_BRANCH}"
    fi

    git branch -D "${TMP_BRANCH}" 
  fi
done

# Go back to original branch
git checkout ${original_branch}

# Go back to orignal folder
popd
