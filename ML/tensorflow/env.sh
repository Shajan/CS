# Python environment setup
# Add to ~/.local.bash
# alias tf.env='source <path-to-this-script>/env.sh'

SCRIPT_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_FOLDER}/env.vars.sh $1
source ${TF_BASE}/${TF_ENV}/bin/activate
