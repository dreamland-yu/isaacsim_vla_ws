#!/bin/bash

# ================================
# Resolve home directory paths
# ================================

USD_PATH="$HOME/isaacsim_vla_ws/config/vla_so101_new_calib_2cam.usd"
ISAACSIM_ROOT="/isaacsim"

# ================================
# Start Isaac Sim with USD + layout
# ================================

cd "$ISAACSIM_ROOT"

./isaac-sim.sh --exec "open_stage.py $USD_PATH"
