#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper to execute the requested CLI inside the container.
exec "$@"
