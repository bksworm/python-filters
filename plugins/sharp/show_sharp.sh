#!/bin/sh


rm -rf  ~/.cache/gstreamer-1.0

GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD:$HOME/.local/lib/x86_64-linux-gnu/gstreamer-1.0 \
GST_DEBUG=python:6 gst-launch-1.0 -v filesrc location="$1" ! matroskademux  \
	! h264parse ! openh264dec ! videorate ! video/x-raw,framerate=10/1 \
    ! tee name=t ! queue ! videocrop bottom=540 ! videoconvert ! gtksink \
	t. !  queue ! videocrop top=960 right=560 ! videoconvert ! video/x-raw,format=GRAY8  \
	! uv_sharp_py min=5 contrast=false \
    ! videocrop right=540 bottom=405 ! videoconvert ! gtksink

