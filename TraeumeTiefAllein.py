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

if __name__ == '__main__':
	cnn = deep_dream.CNN(deep_dream.GOOGLENET_BVLC, cpu_workers=16)
	input_img = Image.open('Klassentreffen_Bilder2.jpg').resize((1200, 1600), Image.LANCZOS)

	# input_img.save("./Meine_Bilder/out.png", "png")

	output_img = cnn.dream(input_img, {'inception_4a/pool_proj': 1}, scales=7*4, n=2, per_octave=8, step_size=4)

	# output_img.save("./Meine_Bilder/Dreamed_out.png", 'png')

	deep_dream.to_image(output_img).save("./Meine_Bilder/Tobias_dreamed_out.jpg", quality=85)

