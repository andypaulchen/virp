#!/bin/bash

# Check for input directory
if [ -z "$1" ]; then
    echo "Usage: $0 <subdirectory>"
    exit 1
fi

basedir="$1"
suffix="_toten_soc"

# Clean up any old temp files
find "$basedir" -name "*${suffix}.dat.tmp" -delete

# Loop over all out_spte files
find "$basedir" -type f -name "$2" | while read -r filepath; do
    calc_dir=$(dirname "$filepath")
    calcname=$(basename "$calc_dir")
    subfolder_dir=$(dirname "$calc_dir")
    subfolder=$(basename "$subfolder_dir")
    outfile="${subfolder_dir}/${subfolder}${suffix}.dat"

    # Extract final energy line
    energy=$(grep "F=" "$filepath" | tail -n1 | awk '{for (i=1;i<=NF;i++) if ($i ~ /^F=/) print $(i+1)}')

    # Temporarily store results
    echo -e "${calcname}\t${energy}" >> "${outfile}.tmp"
done

# Overwrite each _toten.dat with sorted output (version sort handles numeric suffixes)
find "$basedir" -name "*${suffix}.dat.tmp" | while read -r tmpfile; do
    outfile="${tmpfile%.tmp}"
    sort -V "$tmpfile" > "$outfile"
    rm "$tmpfile"
done
