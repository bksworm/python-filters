#!/bin/sh

#GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
#GST_DEBUG=*:3,python:6 \
#gst-launch-1.0 -v -e  videotestsrc num-buffers=120 ! video/x-raw,width=640,height=480,format=I420 \
	#! uv_clip_py min=9 max=255 \
	#! videoconvert ! gtksink
    #1280 -720 
rm -rf  ~/.cache/gstreamer-1.0

GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
GST_DEBUG=*:2,python:6 gst-launch-1.0 -v filesrc location="$1" ! matroskademux  \
	! h264parse ! openh264dec ! videorate ! video/x-raw,framerate=10/1 \
    ! tee name=t ! queue ! videocrop bottom=540 ! videoconvert ! gtksink \
	t. !  queue ! videocrop top=960 right=560 ! videoconvert ! video/x-raw,format=GRAY8  \
	! uv_conv2d_py min=10 kernel-size=4 ! videoconvert ! gtksink
