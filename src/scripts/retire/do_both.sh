function preprocess () {
    # preprocess data
    
    MOVIEDIR=$1
    GDSTRTPT_LOC=$2
    FINAL_DIR=$3
    FINAL_CSV=$4

    echo "[*] extracting frames from movies --> frames/";
    python src/scripts/extract_frames.py $GDSTRTPT_LOC $MOVIEDIR frames/;
    
    echo "[*] partitioning trials ---> partitions/, partitions.csv";
    python src/scripts/partition_trials.py frames/ partitions/ partitions.csv;
    
    echo "[*] resizing trials ---> rsz/";
    python src/scripts/resize_trials.py partitions/ rsz/
    
    echo "[*] consolidating partitions ---> consolidated/";
    python src/scripts/consolidate_data.py rsz/ $FINAL_DIR
    
    echo "[*] cataloguing partitions ---> catalogue.csv";
    python src/scripts/catalog_partitions.py partitions/ DeepLearningClassData.csv
    
    echo "[*] creating labels csv ---> partitions_cons.csv";
    python src/scripts/consolidated_to_csv.py partitions_cons.csv $FINAL_CSV
}

function cleanup () {
    # clean up the artifacts of preprocessing

    if [ -d frames/ ]; then
        echo "[-] rm -r frames"
        rm -r frames
    fi
    
    if [ -d partitions/ ]; then
        echo "[-] rm -r partitions/"
        rm -r partitions
    fi
    
    if [ -d rsz/ ]; then
        echo "[-] rm -r rsz/"
        rm -r rsz
    fi
    
    if [ -f partitions.csv ]; then
        echo "[-] rm partitions.csv"
        rm partitions.csv
    fi
    
    if [ -f partitions_cons.csv ]; then
        echo "[-] rm partitions_cons.csv"
        rm partitions_cons.csv
    fi
}

function control_c () {
    # on keyboard interrupt

    cleanup
    
    if [ -d aug_consolidated ]; then
        echo "[-] rm -r aug_consolidated"
        rm -r aug_consolidated 
    fi

    if [ -d reg_consolidated ]; then
        echo "[-] rm -r reg_consolidated"
        rm -r reg_consolidated 
    fi
    
    if [ -f reg_part_out.csv ]; then
        echo "[-] rm reg_part_out.csv"
        rm reg_part_out.csv
    fi


    if [ -f aug_part_out.csv ]; then
        echo "[-] rm reg_part_out.csv"
        rm aug_part_out.csv
    fi

    kill $PID
    exit
}

trap control_c SIGINT

if [ ! -d augmented/ ]; then
    echo "[+] augmenting movie selection ... "
    python src/scripts/augment_speed.py data/ NextStartingPoint.csv augmented/   
fi 

preprocess data NextStartingPoint.csv reg_consolidated reg_part_out.csv
cleanup
preprocess augmented NextStartingPoint.csv aug_consolidated aug_part_out.csv
cleanup
