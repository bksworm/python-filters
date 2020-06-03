#!/bin/sh

GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD \
GST_DEBUG=*:3,python:6 gst-launch-1.0 -v \
    mpegtsmux name=mixer ! filesink location=clip10conv4x4-contrast-2str.ts \
    filesrc location="$1" ! matroskademux  \
	! h264parse ! openh264dec ! videorate ! video/x-raw,framerate=10/1  \
    ! tee name=t ! queue ! videocrop bottom=540 ! videoscale ! video/x-raw,width=640,height=480 \
    ! x265enc ! mixer.sink_00 \
	t. !  queue ! videocrop top=960 right=560 \
    ! videoconvert ! video/x-raw,format=GRAY8 \
	! uv_conv2d_py min=10 kernel-size=4 contrast=true \
    ! videocrop right=540 bottom=405 ! videoscale ! video/x-raw,width=640,height=480 \
    ! videoconvert ! video/x-raw,format=I420 ! x265enc ! mixer.sink_01

#    ! x264enc ! mp4mux ! filesink location=test.mp4 \
