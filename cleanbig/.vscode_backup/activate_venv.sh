#!/bin/bash
# Source the venv automatically
source "${PWD}/ecom_env/bin/activate"
# Then source the normal bashrc to preserve settings
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
