setup_once() {
  virtualenv ./env
  ./env/bin/activate
  pip install django
  django-admin --version
}

# Need to run init before this step
setup_project() {
  django-admin startproject helloworld
  pushd ./helloworld
  python ./manage.py startapp calc
  popd
}

init() {
  echo "run: source ./env/bin/activate"
}


#setup_once
init
