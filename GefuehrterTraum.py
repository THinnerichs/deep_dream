import deep_dream as dd
from PIL import Image


if __name__ == '__main__':
	cnn = dd.CNN(dd.GOOGLENET_BVLC, cpu_workers=64)
	input_img = Image.open('kodim/img0022.jpg').resize((1538, 1024), Image.LANCZOS)

	guide_img = Image.open('example2_guide.jpg').resize((1538, 1024), Image.LANCZOS)

	output_img = cnn.dream_guided(input_img, guide_img, {'inception_4a/pool_proj': 1})
	
	dd.to_image(output_img).save("./Meine_Bilder/standard_example_guided_dream_out.jpg", quality=90)








