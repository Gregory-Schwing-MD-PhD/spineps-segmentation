#!/bin/bash
# SPINEPS wrapper script
# Calls spineps CLI with proper arguments

set -e

# Pass all arguments to spineps
spineps "$@"

exit $?
