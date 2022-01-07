#!/bin/bash
TARGET_DIR=$1
classes="boxing handclapping handwaving jogging running walking"
img_size=64
for class in $classes; do
	for fname in $TARGET_DIR/raw/$class/*; do
		fname="$(basename -- $fname)"
		mkdir -p $TARGET_DIR/processed/$class/${fname:0:-11}
		ffmpeg -i $TARGET_DIR/raw/$class/${fname} -r 25 -f image2 -s ${img_size}x${img_size} $TARGET_DIR/processed/$class/${fname:0:-11}/image-%03d_${img_size}x${img_size}.png
	done
done
