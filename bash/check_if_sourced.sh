SCRIPT_NAME=${BASH_SOURCE[0]}

[[ $_ != $0 ]] && echo "run: source ${SCRIPT_NAME}" || echo "sourcing ${SCRIPT_NAME}"

export FOO=BAR
echo "Setting env FOO=${FOO}"
