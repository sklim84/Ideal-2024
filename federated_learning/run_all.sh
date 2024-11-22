#!/bin/bash

# Default values for num_samples_org and num_samples_syn

# Parse command-line arguments
while getopts "o:s:" opt; do
  case $opt in
    o) num_samples_org=$OPTARG ;;
    s) num_samples_syn=$OPTARG ;;
    *) echo "Usage: $0 [-o num_samples_org] [-s num_samples_syn]" >&2; exit 1 ;;
  esac
done

# List of Python scripts to run
scripts=(
    "2_make_syn_data_by_fl_avg_ctgan.py"
    "2_make_syn_data_by_fl_avg_tvae.py"
    "2_make_syn_data_by_fl_sgd_ctgan.py"
    "2_make_syn_data_by_to_ctgan.py"
    "2_make_syn_data_by_to_tvae.py"
    "2_make_syn_data_by_lo_ctgan.py"
    "2_make_syn_data_by_lo_tvae.py"
)

# Execute each script with arguments
for script in "${scripts[@]}"; do
    echo "Running $script with num_samples_org=$num_samples_org and num_samples_syn=$num_samples_syn..."
    python3 "$script" --num_samples_org "$num_samples_org" --num_samples_syn "$num_samples_syn"

    # Check if the last script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error encountered while running $script. Exiting..."
        exit 1
    fi
done

