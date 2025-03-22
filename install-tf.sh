#!/bin/bash
set -x

pip install "tensorflow[and-cuda]>=2.17.0" tf_keras
pip install -e "git+https://github.com/askerlee/deepface@master#egg=deepface"
