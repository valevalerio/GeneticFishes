#!/bin/bash
echo $1 $2
folder=$1

cd $folder
rm *.mp4
#get the genertion recorded
list=`ls *.jpg | grep -o Generation_[0-9]*\_ | grep -o  [0-9][0-9][0-9] | sort -u`

echo $list

for f in $list; 
	do echo $f;
	ffmpeg -framerate 25 -i Generation_$f\_time_%03d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output$f.mp4
done
rm mylist.txt
for f in $list; 
	do echo "file 'output$f.mp4'">>mylist.txt;
done

ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4 
rm output[0-9][0-9][0-9].mp4
cd ..

mv $folder/output.mp4 ./output_from_$folder.mp4


