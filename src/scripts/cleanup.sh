# remove artifacts

echo "[!] removing artifacts"

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
