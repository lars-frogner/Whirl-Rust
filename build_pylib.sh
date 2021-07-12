#!/bin/sh
cd python-whirl && maturin build --release && cd - && pip uninstall -y python-whirl && pip install target/wheels/python_whirl-0.1.0-cp36-cp36m-macosx_10_7_x86_64.whl
