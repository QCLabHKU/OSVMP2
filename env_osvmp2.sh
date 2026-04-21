#!/bin/bash

# Check if we are in Zsh or Bash/other
if [ -n "$ZSH_VERSION" ]; then
    # Zsh specific way to get the script's directory
    SCRIPT_PATH="${(%):-%x}"
else
    # Bash and others
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi

# Get the absolute directory path
export OSVPATH=$(cd -- "$(dirname -- "$SCRIPT_PATH")" &> /dev/null && pwd)
export PYTHONPATH=$OSVPATH:$PYTHONPATH
export osvmp2=$OSVPATH/osvmp2/opt_df.py