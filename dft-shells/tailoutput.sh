#!/bin/bash

# Checks the last few rows of output files
# Usage: ./tailoutput.sh system_name

system="$1"
outputfilename="$2"

if [[ -z "$system" ]]; then
    echo "Usage: $0 [system]"
    exit 1
fi

if [[ ! -d "$system" ]]; then
    echo "Error: Directory '$system' not found."
    exit 1
fi

# Loop over only subdirectories in the specified system
for runfolder in "$system"/*; do
    if [[ ! -d "$runfolder" ]]; then
        continue
    fi

    outfile="$runfolder/$outputfilename"

    if [[ ! -f "$outfile" ]]; then
        echo "Missing file: $outfile"
        all_ok=false
        continue
    fi

    tail $outfile
done
