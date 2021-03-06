#!/bin/sh

GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
GST_DEBUG=*:3,python:6 gst-launch-1.0 -v \
    compositor name=mixer background=black width=1280 height=1500 sink_1::xpos=0 sink_1::ypos=960  \
    ! x265enc ! mpegtsmux ! filesink location=clip10conv4x4-contrast.ts \
    filesrc location="$1" ! matroskademux  \
	! h264parse ! openh264dec ! videorate ! video/x-raw,framerate=10/1  \
    ! tee name=t ! queue ! videocrop bottom=540 ! video/x-raw,format=I420,width=1280,height=960 ! mixer. \
	t. !  queue ! videocrop top=960 right=560 ! videoconvert ! video/x-raw,format=GRAY8 \
	! uv_conv2d_py min=10 kernel-size=4 contrast=true ! videocrop right=540 bottom=405 \
    ! videoscale ! video/x-raw,width=720,height=540 \
    ! videoconvert ! video/x-raw,format=I420 ! mixer.

#    ! x264enc ! mp4mux ! filesink location=test.mp4 \
