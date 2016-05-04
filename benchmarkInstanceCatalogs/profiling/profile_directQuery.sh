#!/usr/bin/env bash
# usage: 
# ./"$1" script_to_profile statisticsFile outfileFromcProfile
outfile='prof'
statsfile='stats.txt'

echo $outfile
if [[ "$#" -ne 0 ]]; then
    echo 'change name'
    script="$1"
    if [[ "$#" -ne 1 ]]; then
    statsfile="$2"
    outfile="$3"
    fi
fi
echo $#
echo $outfile

python -m cProfile -o $outfile $script
python dumpprof.py $outfile $statsfile 
