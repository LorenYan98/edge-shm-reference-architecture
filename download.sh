#!/bin/bash
# Public datasets for the edge-SHM experiment. Re-runnable; curl -C - resumes.
set -u
cd "$(dirname "$0")/raw"
get() { url="$1"; out="$2"; [ -s "$out" ] && [ -z "$(curl -sI "$url" | grep -i '^content-length' | awk '{print $2}' | tr -d '\r' | awk -v s=$(stat -c%s "$out") '$1!=s')" ] && { echo "have $out"; return; }; curl -sL -C - --retry 5 -o "$out" "$url" && echo "done $out $(stat -c%s "$out")"; }
Z=https://zenodo.org/records
get $Z/3745914/files/matlab-functions.zip   kw51_matlab-functions.zip
get $Z/3745914/files/trackedmodes.zip       kw51_trackedmodes.zip
get $Z/8300495/files/DiB.zip                vanersborg_DiB.zip
get $Z/3745914/files/ambient_201901.zip     kw51_ambient_201901.zip
get $Z/3745914/files/ambient_202001.zip     kw51_ambient_202001.zip
