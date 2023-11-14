#!/bin/bash

# Path to the virtual environment directory
VENV=web_server
VENV_PATH="`pwd`/${VENV}"
REQUIREMENTS=("flask")

export PORT=8080

is_installed() {
  PACKAGE="${1}"
  python -c "import ${PACKAGE}" && echo "yes" || echo "no"
}

set_env() {
  # Use virtual environment
  if [[ "${VIRTUAL_ENV}" != "" && "${VIRTUAL_ENV}" != "${VENV_PATH}" ]]
  then
    echo "Error: wrong env ${VIRTUAL_ENV}, expect ${VENV_PATH}."
    exit 1
  elif [[ "${VIRTUAL_ENV}" == "" ]]
  then
    source "${VENV_PATH}/bin/activate"
  fi
}

init() {
  # Setup virtual environment
  if [ ! -d "${VENV_PATH}" ]
  then
    python3 -m venv "${VENV}"
  fi

  set_env

  # Make sure packages are installed
  for package in "${REQUIREMENTS[@]}"
  do
    installed=$(is_installed "${package}")
    if [ "${installed}" == "no" ]
    then
      pip install "${package}"
    fi
  done
}

start_server() {
  python ./server.py -p "${PORT}"
}

#init

set_env
start_server
