#!/bin/bash

# Usage: ./checkrelag.sh system_name

system="$1"
outputfilename="out_sprelag"

if [[ -z "$system" ]]; then
    echo "Usage: $0 [system]"
    exit 1
fi

if [[ ! -d "$system" ]]; then
    echo "Error: Directory '$system' not found."
    exit 1
fi

# Flag to track whether all relaxations are complete
all_ok=true

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

    # Get the second to last line
    second_last_line=$(tail -n 2 "$outfile" | head -n 1)

    required_line=" reached required accuracy - stopping structural energy minimisation"

    if [[ "$second_last_line" != "$required_line" ]]; then
        echo "$(basename "$runfolder")"
        all_ok=false
    fi
done

# Final status message
if $all_ok; then
    echo "All relaxations complete."
fi

