USER_NAME=yangyunlun

docker build --pull --no-cache -t $USER_NAME/tensorflow-serving-devel-gpu \
  -f tensorflow_serving/tools/docker/Dockerfile.devel-gpu .
docker build -t $USER_NAME/tensorflow-serving-gpu \
  --build-arg TF_SERVING_BUILD_IMAGE=$USER_NAME/tensorflow-serving-devel-gpu \
  -f tensorflow_serving/tools/docker/Dockerfile.gpu .
