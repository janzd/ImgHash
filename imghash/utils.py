import scipy as sp
import os

def binary_to_hex(binary):
	string = ""
	number = 0 
	modulo = 4
	for i in range(binary.size):
		pos = i % modulo
		number = number + (2 ** pos) * binary[i]
		if pos == modulo-1:		
			hexa = format(number, 'x')
			string += hexa
			number = 0
	return string

def hamming_distance(str1, str2):
	if len(str1) != len(str2):
		raise ValueError("Strings have different sizes!")
	return sum(char1 != char2 for char1, char2 in zip(str1, str2))

def convert_to_grayscale(img):
	shape = img.shape(img)
	if len(shape) == 1:
		return img
	elif len(shape) == 3:
		return sp.inner(img, [299, 587, 114]) / 1000
	elif len(shape) == 4:
		return sp.inner(img, [299, 587, 114, 0] / 1000)
	elif len(shape) == 2:
		return sp.inner(img, [1, 0])
	else:
		raise ValueError("The image has a non-standard bit-depth which is not supported.")

def get_image_filenames(directory=os.curdir):
	image_filenames = []
	for f in os.listdir(directory):
		if f.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".jpg-large")):
			image_filenames.append(f)
	print(image_filenames)
	return image_filenames
