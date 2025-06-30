#!/bin/bash

# Define the subfolder as a variable
SUBFOLDER="$1"
INCARVERSION="INCAR-te-soc"
JOBSUB="te-soc.pbs"

# Optional flag to skip CONTCAR copying
SKIP_CONTCAR_COPY=false
if [ "$2" == "--nocopy" ]; then
    SKIP_CONTCAR_COPY=true
fi


# Check if SUBFOLDER exists
if [ ! -d "$SUBFOLDER" ]; then
    echo "Error: $SUBFOLDER directory does not exist."
    exit 1
fi

# Check if required files exist
if [ ! -f "$INCARVERSION" ] || [ ! -f "$JOBSUB" ] ; then
    echo "Error: One or more required files (INCAR, KPOINTS, te.pbs) not found."
    exit 1
fi

# Process each POSCAR file in SUBFOLDER
for subdir in "$SUBFOLDER"/*/; do
    # Extract the base name without extension
    base_name=$(basename "$subdir")

    # Skip if CONTCAR doesn't exist
    if [ ! -f "$subdir/CONTCAR" ]; then
        echo "Warning: Skipping $base_name (missing CONTCAR)"
        continue
    fi
    
    # Copy required files
    cp $INCARVERSION "$SUBFOLDER/$base_name/INCAR"
    cp "KPOINTS" "$SUBFOLDER/$base_name/"
    
    # Copy CONTCAR to POSCAR and extra copy for keeps (unless skipped)
    if [ "$SKIP_CONTCAR_COPY" = false ]; then
        cp "$SUBFOLDER/$base_name/CONTCAR" "$SUBFOLDER/$base_name/POSCAR"
        cp "$SUBFOLDER/$base_name/CONTCAR" "$SUBFOLDER/contcar-${base_name}.vasp"
    fi

    
    # Create modified .pbs file
    sed "2s/#PBS -N NAME/#PBS -N ${base_name}/" $JOBSUB > "$SUBFOLDER/$base_name/$JOBSUB"
    
    # Submit job
    cd "$SUBFOLDER/$base_name"
    qsub $JOBSUB
    cd ../../
   
    echo "Processed $base_name"
done

echo "All jobs submitted."
