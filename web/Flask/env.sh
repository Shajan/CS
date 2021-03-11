setup_once() {
  virtualenv ./env
  ./env/bin/activate
  pip3 install flask
}

init() {
  echo "run: source ./env/bin/activate"
}

#setup_once
init
