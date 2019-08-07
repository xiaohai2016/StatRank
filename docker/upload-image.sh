#docker pull pytorch/pytorch
docker build ./ -t pytorch/mypytorch
docker tag pytorch/mypytorch docker.artifactory-test.corp.linkedin.com/tensorflow/xiaozhan-pytorch:0.0.6
docker push docker.artifactory-test.corp.linkedin.com/tensorflow/xiaozhan-pytorch:0.0.6
