K8S_CLUSTER=minikube

install_client() {
  brew install kubectl
}

setup_k8s_on_mac() {
  brew install minikube
}

start_on_mac() {
  minikube config set driver docker
  minikube start --driver=docker
}

init_gke() {
  gcloud auth login
  gcloud config set project ${PROJECT_ID}
}

init_kubectl() {
  kubectl config use-context ${K8S_CLUSTER}
  kubectl config current-context
}

dashboard() {
  minikube dashboard
}

#install_client

# Next two steps if running minikube on mac
#setup_k8s_on_mac
#init_mac
#init_kubectl
#dashboard
