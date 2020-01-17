import io

import numpy as np
from PIL import Image

import deep_dream

def save_image(img, name):
	
	im = np.array(img)
	fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(im)))

	visual = np.log(fft_mag)
	visual = (visual - visual.min()) / (visual.max() - visual.min())

	result = Image.fromarray((visual * 255).astype(np.uint8))
	result.save(name+'.bmp')

cnn = deep_dream.CNN(deep_dream.GOOGLENET_BVLC, cpu_workers=0, gpus=[0])
input_img = Image.open('kodim/img0022.jpg').resize((1536, 1024), Image.LANCZOS)
# save_image(input_img, './Meine_Bilder/out')
input_img.save("./Meine_Bilder/out.png", "png")

