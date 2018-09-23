#!/bin/sh

# $1 is the relative path to the chosen image.
# $2 is the name of the class that should be cut out
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters."
    exit 1
fi

python3 masks.py $1

img_basename=$(basename -- $1)
img_name="${img_basename%.*}"
inpaint_path=../generative_inpainting

echo "Original image path: $1"
echo "Generated mask path ($2): ../images/results/masks/$img_name/$2.jpg"
echo "Result image path: $inpaint_path/examples/$img_name.png"

python3 $inpaint_path/test.py --image $1 --mask ../images/results/masks/$img_name/$2.jpg \
	--output $inpaint_path/examples/$img_name_$2.png --checkpoint_dir $inpaint_path/model_logs/release_imagenet_256

