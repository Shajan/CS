PROJECT_ID=my-gcp-project
APP=echo
APP_IMAGE=${APP}-image
APP_SERVICE=${APP}-service
TAG=v1
GCR_IMAGE="gcr.io/${PROJECT_ID}/${APP_IMAGE}:${TAG}"
K8S_CLUSTER=minikube

# Help with gcr
#
# https://ahmet.im/blog/google-container-registry-tips/
#  docker search gcr.io/google-containers/kube
#  gcloud container images list --repository=${GCR_IMAGE}
#
# https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
#  kubectl scale deployment ${APP} --replicas=2 
#  kubectl autoscale deployment ${APP} --cpu-percent=80 --min=1 --max=5

init_gke() {
  gcloud auth login
  gcloud config set project ${PROJECT_ID}
}

init_kubectl() {
  kubectl config use-context ${K8S_CLUSTER}
}

build_for_gke() {
  docker build -t ${APP_IMAGE} .
  docker tag ${APP_IMAGE} ${GCR_IMAGE}
}

build_for_minikube() {
  # minikube has it's own image registry?
  # minikube docker-env
  eval $(minikube -p minikube docker-env)
  docker build -t ${APP_IMAGE} .
  docker tag ${APP_IMAGE} ${GCR_IMAGE}
}

publish_gcr() {
  docker push ${GCR_IMAGE}
}

verify_gcr() {
  docker pull ${GCR_IMAGE}
}

docker_run_local() {
  docker run --rm -p 80:80 ${GCR_IMAGE}
}

create_gke_deployment() {
  kubectl create deployment ${APP} --image=${GCR_IMAGE}
}

minikube_run() {
  kubectl run ${APP} --image=${APP_IMAGE} --image-pull-policy=Never
}

verify_pods() {
  kubectl get deployments
  kubectl get pods
}

gke_expose_service() {
  kubectl expose deployment ${APP} --name=${APP_SERVICE} --type=LoadBalancer --port 80 --target-port 80
}

minikube_expose_service() {
  # Run 'minikube service echo-service' from a different terminal
  kubectl expose deployment ${APP} --name=${APP_SERVICE} --type=LoadBalancer --port 80 --target-port 80
}

verify_service() {
  kubectl get service
}

gke_send_request() {
  end_point=$(kubectl get service | egrep ${APP_SERVICE} | awk '{print $4}')
  echo "curl ${end_point}"
  curl ${end_point}
}

minikube_send_request() {
  minikube service ${APP_SERVICE}
}

remove_deployment() {
  kubectl delete deployment ${APP}
}

cleanup() {
  gcloud container images delete ${GCR_IMAGE} --force-delete-tags
  docker rmi $(docker images -q)
}

#################################################
# Helpful minikube commands
#
# minikube ssh 
#   docker images
# minikube dashboard
#
# To point docker commands to minikube's docker
# eval $(minikube docker-env)    # Set env
# eval $(minikube docker-env -u) # Uset env
#
# Setting up tunnel
# minikube service <service-name>
# minikube tunnel --cleanup
#################################################

#init_kubectl
#build_for_minikube
#gcr_publish
#gcr_verify
#docker_run_local
#minikube_run
#verify_pods
#minikube_expose_service
#verify_service
#send_request
#remove_deployment
#cleanup
