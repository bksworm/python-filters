#import math
#import typing as typ

import torch
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo  # noqa:F401,F402              

from gstreamer import utils
from gstreamer.gst_hacks import map_gst_buffer

BITS_PER_BYTE = 8 

_TORCH_DTYPES = {
    16: torch.int16,
}

def get_tensor_dtype(fmt: GstVideo.VideoFormat) -> np.number:
    format_info = GstVideo.VideoFormat.get_info(fmt)
    return _TORCH_DTYPES.get(format_info.bits, torch.uint8)


_NP_DTYPES = {
    16: np.int16,
}


def get_np_dtype(fmt: GstVideo.VideoFormat) -> np.number:
    format_info = GstVideo.VideoFormat.get_info(fmt)
    return _NP_DTYPES.get(format_info.bits, np.uint8)


def gst_buffer_for_tensor(buffer: Gst.Buffer, *, width: int, height: int, 
                          channels: int,
                          dtype: np.dtype, bpp: int = 1) ->  np.ndarray:
    """Converts Gst.Buffer with known format (w, h, dtype) to np.ndarray"""
    with map_gst_buffer(buffer, Gst.MapFlags.READ) as mapped:
        result = np.ndarray((buffer.get_size() // (bpp // BITS_PER_BYTE)),
                            buffer=mapped, dtype=dtype)
        #make (c,h,w) array from memory buf
        if channels > 0 :
            result = result.reshape(channels, height, width)
        else : #make (c,w,h) array form memory buf   
            result = result.reshape(height,width)
        return result


def gst_buffer_with_pad_for_tensor(buffer: Gst.Buffer, pad: Gst.Pad) -> np.ndarray:
    """Converts Gst.Buffer with Gst.Pad (stores Gst.Caps) to np.ndarray """
    return gst_buffer_with_caps_to_tensor(buffer, pad.get_current_caps())


def gst_buffer_with_caps_for_tensor(buffer: Gst.Buffer, caps: Gst.Caps) -> np.ndarray:
    """ Converts Gst.Buffer with Gst.Caps (stores buffer info) to np.ndarray """

    structure = caps.get_structure(0)  # Gst.Structure

    width, height = structure.get_value("width"), structure.get_value("height")

    # GstVideo.VideoFormat
    video_format = utils.gst_video_format_from_string(structure.get_value('format'))

    channels = utils.get_num_channels(video_format)

    dtype = get_np_dtype(video_format)  # np.dtype

    format_info = GstVideo.VideoFormat.get_info(video_format)  # GstVideo.VideoFormatInfo

    return gst_buffer_for_tensor(buffer, width=width, height=height, 
                                 channels=channels,
                                 dtype=dtype, bpp=format_info.bits)


