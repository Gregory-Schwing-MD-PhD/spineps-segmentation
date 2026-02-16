#!/bin/bash
# Build SPINEPS Docker container

DOCKER_USERNAME="go2432"
IMAGE_NAME="spineps-segmentation"
TAG="latest"

echo "Building SPINEPS container..."
docker build -f Dockerfile.spineps -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} .

echo ""
echo "Build complete!"
echo ""
echo "To push to Docker Hub:"
echo "  docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To run:"
echo "  docker run -it --gpus all ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
