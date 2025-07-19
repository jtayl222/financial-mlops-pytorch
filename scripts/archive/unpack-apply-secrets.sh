
kubectl create ns seldon-system
kubectl create ns seldon-system
mkdir -p k8s/manifests/seldon-system 
mkdir -p k8s/manifests/seldon-system-ml
tar xzf seldon-system-ml-secrets-20250710.tar.gz -C k8s/manifests/seldon-system 
tar xzf seldon-system-ml-secrets-20250708.tar.gz -C k8s/manifests/seldon-system-ml 
kubectl apply -k k8s/manifests/seldon-system-ml/production/
kubectl apply -k k8s/manifests/seldon-system/production/