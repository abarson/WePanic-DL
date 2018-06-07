# preprocess the data

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

# ./cleanup.sh
