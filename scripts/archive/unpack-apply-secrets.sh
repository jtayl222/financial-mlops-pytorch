
kubectl create ns financial-inference
kubectl create ns financial-mlops-pytorch
mkdir -p k8s/manifests/financial-inference 
mkdir -p k8s/manifests/financial-mlops-pytorch-ml
tar xzf financial-inference-ml-secrets-20250710.tar.gz -C k8s/manifests/financial-inference 
tar xzf financial-mlops-pytorch-ml-secrets-20250708.tar.gz -C k8s/manifests/financial-mlops-pytorch-ml 
kubectl apply -k k8s/manifests/financial-mlops-pytorch-ml/production/
kubectl apply -k k8s/manifests/financial-inference/production/