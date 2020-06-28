"""
    Plugin blurs incoming buffer

    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD

    gst-launch-1.0 videotestsrc num-buffers=3 ! uv_clip_py min=9 max=255 ! videoconvert ! gtksink
"""

import logging
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from skimage import exposure
from skimage.util import img_as_float, img_as_ubyte

from gstreamer import Gst, GObject, GLib, GstBase
from mxnet_utils import gst_buffer_with_caps_for_ndarray


FORMATS = "{GRAY8}"

class GstUvMxnetConv2d(GstBase.BaseTransform):

    GST_PLUGIN_NAME = 'uv_mxnet_py'

    __gstmetadata__ = ("mxnet conv2d filter",  # Name
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

        super(GstUvMxnetConv2d, self).__init__()
        self.minimal = 0
        self.kernel_size = 2
        self.contrast = False
        #kernel (NDArray): convolution's kernel parameters.
        self.kernel = None
        #conv (Block): convolutional layer.
        self.conv = None
        # zeros_vec will be activated as soon we new conv2d
        #output tensor size
        self.zeros_vec = None
        #self.set_conv(self.kernel_size) #set def kernel sum2x2

    def set_conv(self, kernel_size:int ):
        self.kernel_size = kernel_size
        self.kernel = nd.ones((kernel_size, kernel_size), dtype=np.float32)
        h, w = self.kernel.shape
        stride=kernel_size
        padding=(kernel_size+1)//2

        self.conv = nn.Conv2D(channels=1, kernel_size=(h,w), 
            strides=(stride, stride), padding=(padding,padding), 
            use_bias=False)

        # add dimensions for channels and in_channels if necessary
        while self.kernel.ndim < len(self.conv.weight.shape):
            self.kernel = self.kernel.expand_dims(0)
        # check if transpose convolution
        if type(self.conv).__name__.endswith("Transpose"):
            in_channel_idx = 0
        else:
            in_channel_idx = 1
        # initialize and set weight
        self.conv._in_channels = self.kernel.shape[in_channel_idx]
        self.conv.initialize()
        self.conv.weight.set_data(self.kernel)


    def apply_conv(self, data: nd.array) -> nd.array:   
        """
        Args:
            data (NDArray): input data.
        Returns:
            NDArray: output data (after applying convolution).
        """
        # add dimensions for batch and channels if necessary
        while data.ndim < len(self.conv.weight.shape):
            data = data.expand_dims(0)
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
            self.set_conv(value)
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
            image = gst_buffer_with_caps_for_ndarray(buffer, caps)
            c, h, w  = image.shape
            #convert to NDArray
            t = nd.array(image)
            #clip low level signals to 0
            if self.minimal > 0:
                if self.zeros_vec is None:
                    #since we don't expect change of rame parametors,
                    # must not make it every time
                  self.zeros_vec = nd.zeros_like(t)
                t = nd.where( t< self.minimal, self.zeros_vec, t)

            t = self.apply_conv(t)
            #clear output
            image.fill(0) 
            #detach from mxnet context and copy to numpy.nddarray[w,h]
            conved = t[0,0, :, :].asnumpy()

            #simple contrast
            if self.contrast == True :
              conved = simpleContrast(conved,0.0)

            #copy to top left coner
            ch, cw= conved.shape 
            image[0, :ch,:cw] = conved
        except Exception as e:
            logging.info(conved.shape)
            logging.error(e)

        return Gst.FlowReturn.OK



# Required for registering plugin dynamically
# Explained:
# http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstUvMxnetConv2d)
__gstelementfactory__ = (GstUvMxnetConv2d.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstUvMxnetConv2d)

def adaptiveEqualization(img: np.ndarray, clip_limit:float =0.03) ->  np.ndarray:
  img_f = img_as_float(img)
  logging.info(img_f[0, :4])
  return img_as_ubyte(exposure.equalize_adapthist(img_f, clip_limit))


def simpleContrast(img: np.ndarray, skipPart) ->  np.ndarray:
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

def autoMinMax(img, skipPart=0.05):
  line=np.sort(img.flatten())
  return line[round(img.size * skipPart)],line[-round(img.size * skipPart)]

def autoContract(img, skipPart):
  img_f = img.astype("float64")
  minVal, maxVal = autoMinMax(img_f, skipPart)
  out_img_f = (img-minVal)*(255.0/(maxVal-minVal))
  out_img_f = np.clip(out_img_f, 0.0, 255.0)
  out_img = out_img_f.astype("uint8")
  return out_img
