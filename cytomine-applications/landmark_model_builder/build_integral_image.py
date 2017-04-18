import numpy as np
import scipy.misc as misc
from multiprocessing import Pool


def build_integral_image_from_path(path_to_image):
	img = misc.imread(path_to_image,flatten=True)
	img = img.astype(float)/255.
	i_img = build_integral_image(img)
	
	for ext in ['bmp','png','jpg']:
		path_to_image = path_to_image.rstrip('.%s'%ext)
	path_to_image += '_integral'
	np.save(path_to_image,i_img)


def build_integral_image(img):
	(h,w) = img.shape
	i_img = np.zeros((h,w))
	
	i_img[0,0] = img[0,0]
	for i in range(1,h):
		i_img[i,0] = i_img[i-1,0]+img[i,0]
	for i in range(1,w):
		i_img[0,i] = i_img[0,i-1]+img[0,i]
	for i in range(1,h):
		for j in range(1,w):
			i_img[i,j] = img[i,j]+i_img[i-1,j]+i_img[i,j-1]-i_img[i-1,j-1]
	return i_img


def build_integral_images_mp(tab_path,n_jobs):
	p = Pool(n_jobs)
	p.map(build_integral_image_from_path,tab_path)
	p.close()
	p.join()


if __name__ == "__main__":
	paths = ['/home/remy/datasets/bigres/%3.3d.bmp'%i for i in range(1,101)]
	build_integral_images_mp(paths,4)
