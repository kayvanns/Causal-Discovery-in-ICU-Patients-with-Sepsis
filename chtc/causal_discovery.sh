#!/bin/bash
echo "Starting run $1 ($2)..."
python causal_discovery_fisherz.py --run_index $1 --run_name $2 --data_path $3
echo "Done."