#!/bin/bash
#
# SPINEPS WRAPPER
# Calls SPINEPS via Python module - the CLI binary is broken in the Docker image
#
set -e

export SPINEPS_SEGMENTOR_MODELS=${SPINEPS_SEGMENTOR_MODELS:-/app/models}
export SPINEPS_ENVIRONMENT_DIR=${SPINEPS_ENVIRONMENT_DIR:-/app/models}

mkdir -p "${SPINEPS_SEGMENTOR_MODELS}"

# CRITICAL: must call python -m spineps.entrypoint, NOT the 'spineps' binary
exec python -m spineps.entrypoint "$@"
