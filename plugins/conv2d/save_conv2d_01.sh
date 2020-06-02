#!/bin/sh

#GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
#GST_DEBUG=*:3,python:6 \
#gst-launch-1.0 -v -e  videotestsrc num-buffers=120 ! video/x-raw,width=640,height=480,format=I420 \
	#! uv_clip_py min=9 max=255 \
	#! videoconvert ! gtksink
#1280 -720 

GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
GST_DEBUG=*:3,python:6 gst-launch-1.0 -v \
    compositor name=mixer background=black width=1280 height=1500 sink_1::xpos=0 sink_1::ypos=960  \
    ! x265enc ! mpegtsmux ! filesink location=clip10conv2x2.ts \
    filesrc location="$1" ! matroskademux  \
	! h264parse ! openh264dec ! videorate ! video/x-raw,framerate=10/1  \
    ! tee name=t ! queue ! videocrop bottom=540 ! video/x-raw,format=I420,width=1280,height=960 ! mixer. \
	t. !  queue ! videocrop top=960 right=560 \
    ! videoconvert ! video/x-raw,format=GRAY8 \
	! uv_conv2d_py min=10 \
    ! videoconvert ! video/x-raw,format=I420,width=720,height=540 ! mixer.

#    ! x264enc ! mp4mux ! filesink location=test.mp4 \
