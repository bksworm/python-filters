"""
    Plugin blurs incoming buffer

    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD

    gst-launch-1.0 videotestsrc num-buffers=3 ! uv_clip_py min=9 max=255 ! videoconvert ! gtksink
"""

import logging
import numpy as np
from torch import nn
import torch

from gstreamer import Gst, GObject, GLib, GstBase
from torch_utils import gst_buffer_with_caps_for_tensor


FORMATS = "{GRAY8}"

class GstUvConv2d(GstBase.BaseTransform):

    GST_PLUGIN_NAME = 'uv_conv2d_py'

    __gstmetadata__ = ("Torch con2d filter",  # Name
                       "Filter",   # Transform
                       "Apply con2d filter to Buffer",  # Description
                       "k10")  # Author

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                            Gst.PadDirection.SRC,
                            Gst.PadPresence.ALWAYS,
                            Gst.Caps.from_string(f"video/x-raw,format={FORMATS}")),
                        Gst.PadTemplate.new("sink",
                            Gst.PadDirection.SINK,
                            Gst.PadPresence.ALWAYS,
                            Gst.Caps.from_string(f"video/x-raw,format={FORMATS}"))
                        )
    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        # Parameters from clip
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#gaussianblur
        "min": (GObject.TYPE_INT64,  # type
                   "Minimum value",  # nick
                   "Clip Minimum value",  # blurb
                   0,  # min
                   254,  # max
                   0,  # default
                   GObject.ParamFlags.READWRITE  # flags
                   ),
        "contrast": (GObject.TYPE_BOOLEAN,  # type
                   "Simple contrast ",  # nick
                   "Simple contrast after conv2d",  # blurb
                   False,  # default
                   GObject.ParamFlags.READWRITE  # flags
                   ),
        "kernel-size": (GObject.TYPE_INT64,  # type
                   "Cov2d kernel size",  # nick
                   "Set Cov2d kernel size and stride kernel_size/2+1",  # blurb
                   1,  # min
                   12,  # max
                   2,  # default
                   GObject.ParamFlags.READWRITE  # flags
                   ),
    }

    def __init__(self):

        super(GstUvConv2d, self).__init__()
        self.minimal = 0
        self.kernel_size = 2
        self.contrast = False
        self.kernel = None
        self.conv = None
        # zeros_vec will be activated as soon we new conv2d
        #output tensor size
        self.zeros_vec = None
        self.set_kernel(self.kernel_size) #set def kernel sum2x2

    def set_kernel(self, kernel_size:int ):
        self.kernel_size = kernel_size
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.uint8)
        h, w = kernel.shape
        self.conv = nn.Conv2d(1, 1, kernel_size=(h,w), 
            stride=kernel_size, padding=kernel_size//2+1, 
            bias=False, padding_mode='zeros')
        self.conv.weight = nn.Parameter(torch.reshape(kernel, (1,1,h,w))
            , requires_grad=False)

    def apply_conv(self, data: torch.tensor) -> torch.tensor:
        return self.conv(data) 

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'min':
            return self.minimal
        elif prop.name == 'kernel-size':
            return self.kernel_size
        elif prop.name == 'contrast':
            return self.contrast
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'min':
            self.minimal = value
        elif prop.name == 'kernel-size':
            self.set_kernel(value)
        elif prop.name == 'contrast':
            self.contrast = value
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        try:
            # convert Gst.Buffer to np.ndarray
            caps = self.sinkpad.get_current_caps()
            #it's a reference on GstBuffer data
            #if you modify it output will be modified as well
            image = gst_buffer_with_caps_for_tensor(buffer, caps)
            c, h, w  = image.shape
            #convert to tensor
            t = torch.tensor(image, dtype=torch.uint8)
            #clip low level signals to 0
            if self.minimal > 0:
                if self.zeros_vec == None:
                    #since we don't expect change of rame parametors,
                    # must not make it every time
                    self.zeros_vec = torch.zeros_like(t)
                t = torch.where( t< self.minimal, self.zeros_vec, t)

            #we have to reshape to pytorch format (N,C,H,W)
            t = torch.reshape( t, (1, c, h, w))
            t = self.apply_conv(t)

            #clear output
            image.fill(0) 
            #detach from pytorch context and copy to numpy.nddarray[w,h]
            conved = t[0,0, :, :].cpu().detach().numpy()

            #simple contrast
            if self.contrast == True :
              conved = simpleContrast(conved)

            #copy to top left coner
            ch, cw= conved.shape 
            image[0, :ch,:cw] = conved
        except Exception as e:
            logging.error(e)

        return Gst.FlowReturn.OK



# Required for registering plugin dynamically
# Explained:
# http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstUvConv2d)
__gstelementfactory__ = (GstUvConv2d.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstUvConv2d)

def simpleContrast(img: np.ndarray) ->  np.ndarray:
  img_min= img.min()
  img_over = (img.max()-img_min)
  img_over = (255 / (img.max()-img_min)) if img_over != 0 else 255

  return ((img - img_min)*img_over).astype("uint8")

def histContrast(img: np.ndarray) ->  np.ndarray :
  h, bin_edges = np.histogram(img, bins=256)
  cdf = np.cumsum(h)
  indxs = np.nonzero(cdf)
  #logging.info(indxs[0][0])
  minCdf = cdf[indxs[0][0]]
  div = (img.size-1.0)
  out_img = (255.0*(cdf[img]-minCdf)/div).astype("ubyte")
  return out_img

def autoMinMax(img, skipPart=0.00):
  line=np.sort(img.flatten())
  return line[round(img.size * skipPart)],line[-round(img.size * skipPart)]

def autoContract(img):
  img_f = img.astype("float64")
  minVal, maxVal = autoMinMax(img_f)
  out_img_f = (img-minVal)*(255.0/(maxVal-minVal))
  out_img_f = np.clip(out_img_f, 0.0, 255.0)
  out_img = out_img_f.astype("uint8")
  return out_img
