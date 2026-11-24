#!/bin/bash
cd "$(dirname "$0")/raw/chillon"
cut -f1,2 ../chillon_list.txt | xargs -P 4 -n 2 sh -c '[ -s "$0" ] || curl -sL --retry 5 -o "$0" "$1"'
echo "chillon done $(ls *.mat | wc -l) files"
