from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)
import fastai
from deoldify.visualize import *
import os
import gradio
import wget
import torch
if not torch.cuda.is_available():
    print('GPU not available.')

COLORIZE_ARTISTIC_LINK = "https://www.dropbox.com/s/zkehq1uwahhbc2o" \
              "/ColorizeArtistic_gen.pth?dl=1"
COLORIZE_ARTISTIC_PATH = './models/ColorizeArtistic_gen.pth'
WATERMARK_LINK = 'https://media.githubusercontent.com/media/jantic/DeOldify' \
                  '/master/resource_images/watermark.png '
WATERMARK_PATH = './resource_images/watermark.png'


if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists(COLORIZE_ARTISTIC_PATH):
    wget.download(COLORIZE_ARTISTIC_LINK, COLORIZE_ARTISTIC_PATH)
if not os.path.exists(WATERMARK_PATH):
    wget.download(WATERMARK_LINK, WATERMARK_PATH)
colorizer = get_image_colorizer(artistic=True)


def predict(inp):
    render_factor = 35
    watermarked = False
    result = colorizer.plot_transformed_image_from_pil(inp,
        render_factor=render_factor, watermarked=watermarked)
    return result


INPUTS = gradio.inputs.Image(cast_to="pillow")
OUTPUTS = gradio.outputs.Image()
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS,
                             load_fn=load)

INTERFACE.launch(inbrowser=True, share=True)
