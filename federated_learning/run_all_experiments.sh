#!/bin/bash

# Define the arguments to pass
num_samples_org=100
num_samples_syn=300

# List of Python scripts to run
scripts=(
    "2_make_syn_data_by_to_tvae.py"
    "2_make_syn_data_by_to_ctgan.py"
    "2_make_syn_data_by_lo_tvae.py"
    "2_make_syn_data_by_lo_ctgan.py"
    "2_make_syn_data_by_fl_sgd_ctgan.py"
    "2_make_syn_data_by_fl_avg_tvae.py"
    "2_make_syn_data_by_fl_avg_ctgan.py"
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

echo "All scripts have been executed successfully."

