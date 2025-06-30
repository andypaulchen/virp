#!/bin/bash

# Define the subfolder as a variable
SUBFOLDER="perov-400"
POTCARFILE="POTCAR_CsSnPbI"
INCARFILE="INCAR-relag"
JOBSUBFILE="relag.pbs"

# Check if SUBFOLDER exists
if [ ! -d "$SUBFOLDER" ]; then
    echo "Error: $SUBFOLDER directory does not exist."
    exit 1
fi

# Check if required files exist
if [ ! -f "INCAR" ] || [ ! -f "KPOINTS" ] || [ ! -f $JOBSUBFILE ] || [ ! -f $POTCARFILE ] ; then
    echo "Error: One or more required files (INCAR, KPOINTS, te.pbs) not found."
    exit 1
fi

# Process each POSCAR file in SUBFOLDER
for poscar_file in $SUBFOLDER/*.vasp; do
    # Extract the base name without extension
    base_name=$(basename "$poscar_file" .vasp)
    
    # Create directory for this POSCAR file
    mkdir -p "$SUBFOLDER/$base_name"
    
    # Copy required files
    cp $INCARFILE "$SUBFOLDER/$base_name/INCAR"
    cp "KPOINTS" "$SUBFOLDER/$base_name/"
    cp $POTCARFILE "$SUBFOLDER/$base_name/POTCAR"
    
    # Copy and rename the VASP file to POSCAR
    cp "$poscar_file" "$SUBFOLDER/$base_name/POSCAR"
    
    # Create modified job submission file
    sed "2s/#PBS -N NAME/#PBS -N ${base_name}/" $JOBSUBFILE > "$SUBFOLDER/$base_name/$JOBSUBFILE"
    
    # Submit job
    cd "$SUBFOLDER/$base_name"
    qsub $JOBSUBFILE
    cd ../../
    
    echo "Processed $base_name"
done

echo "All POSCAR files processed successfully."
