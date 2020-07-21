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
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    render_factor = 35
    watermarked = False
    result = colorizer.plot_transformed_image_from_pil(inp,
        render_factor=render_factor, watermarked=watermarked)
    return result


INPUTS = gradio.inputs.Image()
OUTPUTS = gradio.outputs.Image()
examples = [
    ["examples/mlk.jpg"],
["examples/sinatra-jfk.jpg"],
["examples/payphone.png"]
]
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS, examples=examples,
                             title='De-Oldify', description='De-oldify '
                                                            'colorizes old '
                                                            'images. Try it '
                                                            'out on an old '
                                                            'image you want '
                                                            'restored! Note: It takes about 30s to process the image.', thumbnail='https://i2.wp.com/www.marktechpost.com/wp-content/uploads/2019/08/68747470733a2f2f692e696d6775722e636f6d2f427430766e6b652e6a7067.jpg?fit=1182%2C768&ssl=1')

INTERFACE.launch(inbrowser=True)
