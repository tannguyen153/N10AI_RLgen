#!/bin/bash
module load tensorflow

for dir in *; do
	if [ -d "$dir" ]; then
		echo "cd $dir"
		cd $dir
		python ../../gen_scripts/genRoofline.py
		python ../../gen_scripts/genRoofline_time2D.py
		cd ..
	fi
done
