#!/usr/bin/env bash

echo {0..359} | xargs -n 1 -P 8 python go.py

#cd outs; ffmpeg -r 30 -f image2 -i %d.png -vcodec libvpx-vp9 -crf 25 -b:v 0 ../out.webm
cd outs; ffmpeg -y -r 30 -f image2 -i %d.png -vcodec libx264 -crf 19 -preset veryslow -pix_fmt yuv420p ../out.mp4
#cd outs; ffmpeg -r 30 -f image2 -i %d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p ../out.mp4
