#!/bin/sh


rm -rf  ~/.cache/gstreamer-1.0

GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
GST_DEBUG=*:2,python:6 gst-launch-1.0 -v filesrc location="$1" ! matroskademux  \
	! h264parse ! openh264dec ! videorate ! video/x-raw,framerate=10/1 \
    ! tee name=t ! queue ! videocrop bottom=540 ! videoconvert ! gtksink \
	t. !  queue ! videocrop top=960 right=560 ! videoconvert ! video/x-raw,format=GRAY8  \
	! uv_conv2d_py min=10 kernel-size=4 contrast=true \
    ! videocrop right=540 bottom=405 ! videoconvert ! gtksink

# pylonsrc camera=0  imageformat=ycbcr422_8 width=1280 height=960 offsetx=384 offsety=288 autogain=continuous autoexposure=continuous acquisitionframerateenable=true autobrightnesstarget=0.19608 gainlowerlimit=0 gainupperlimit=36 exposurelowerlimit=50 autowhitebalance=continuous exposureupperlimit=30000 fps=25
# ! video/x-raw,format=YUY2,framerate=25/1  !
#  queue2 ! mixer.
#  pylonsrc camera=1 continuous=false imageformat=mono8 width=720 height=540 autogain=continuous fps=25
#  ! videoconvert ! video/x-raw,format=YUY2  ! queue2 ! mixer.
