import numpy as np
from build_integral_image import *
import sys
from ldmtools import *
from multiprocessing import Pool
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
import scipy.misc as misc
from VotingTreeRegressor import *
"""
HORIZONTAL

-----------------
|XXXXXXX|       |
|XXXXXXX|       |
-----------------

VERTICAL

----------
|XXXXXXXX|
|XXXXXXXX|
|XXXXXXXX|
----------
|        |
|        |
|        |
----------

"""

"""
ATTENTION!!! The number of possibilities increases exponentially with w, so this
function can really really easily jam your computer. Be careful!
"""
def generate_all_2_horizontal(w):
	r = range(-w,w+1)
	w1 = w+1
	return np.array([[x1,y1,x2,y2] for x1 in r for y1 in r for x2 in range(x1+1,w1) for y2 in range(y1,w1)])
	
def generate_2_horizontal(w,n):
	coords = np.zeros((n,4))
	coords[:,0] = np.random.randint(-w,w,n)
	coords[:,1] = np.random.randint(-w,w+1,n)
	coords[:,2] = [coords[i,0]+np.random.randint(1,w+1-coords[i,0]) for i in range(n)]
	coords[:,3] = [coords[i,1]+np.random.randint(0,w+1-coords[i,1]) for i in range(n)]
	return coords
	
def generate_2_vertical(w,n):
	coords = np.zeros((n,4))
	coords[:,0] = np.random.randint(-w,w+1,n)
	coords[:,1] = np.random.randint(-w,w,n)
	coords[:,2] = [coords[i,0]+np.random.randint(0,w+1-coords[i,0]) for i in range(n)]
	coords[:,3] = [coords[i,1]+np.random.randint(1,w+1-coords[i,1]) for i in range(n)]
	return coords
	
def generate_3_horizontal(w,n):
	coords= np.zeros((n,4))
	coords[:,0] = np.random.randint(-w,w-1,n)
	coords[:,1] = np.random.randint(-w,w+1,n)
	coords[:,2] = [coords[i,0]+np.random.randint(2,w+1-coords[i,0]) for i in range(n)]
	coords[:,3] = [coords[i,1]+np.random.randint(0,w+1-coords[i,1]) for i in range(n)]
	return coords

def generate_3_vertical(w,n):
	coords = np.zeros((n,4))
	coords[:,0] = np.random.randint(-w,w+1,n)
	coords[:,1] = np.random.randint(-w,w-1,n)
	coords[:,2] = [coords[i,0]+np.random.randint(0,w+1-coords[i,0]) for i in range(n)]
	coords[:,3] = [coords[i,1]+np.random.randint(2,w+1-coords[i,1]) for i in range(n)]
	return coords

def generate_square(w,n):
	coords = np.zeros((n,4))
	coords[:,0] = np.random.randint(-w,w,n)
	coords[:,1] = np.random.randint(-w,w,n)
	coords[:,2] = [coords[i,0]+np.random.randint(1,w+1-coords[i,0]) for i in range(n)]
	coords[:,3] = [coords[i,1]+np.random.randint(1,w+1-coords[i,1]) for i in range(n)]
	return coords

"""
The signification of the coordinates is given in file ~/Dropbox/lindner/rep.png
"""
def generate_2d_coordinates_horizontal(coords):
	(n,quatre) = coords.shape
	x = np.zeros((n,6))
	y = np.zeros((n,6))
	
	w = np.floor(0.5*((coords[:,2]-coords[:,0])+1.)).astype('int')
	x[:,0] = coords[:,0]-1
	y[:,0] = coords[:,1]-1
	x[:,1] = x[:,0]+w
	y[:,1] = y[:,0]
	x[:,2] = coords[:,2]
	y[:,2] = y[:,1]
	x[:,3] = x[:,0]
	y[:,3] = coords[:,3]
	x[:,4] = x[:,1]
	y[:,4] = y[:,3]
	x[:,5] = x[:,2]
	y[:,5] = y[:,4]
	
	return x.astype('int'),y.astype('int')	

def generate_2d_coordinates_vertical(coords):
	(n,quatre) = coords.shape
	x = np.zeros((n,6))
	y = np.zeros((n,6))
	
	w = np.floor(0.5*((coords[:,3]-coords[:,1])+1)).astype('int')
	x[:,0] = coords[:,0]-1
	y[:,0] = coords[:,1]-1
	x[:,1] = coords[:,2]
	y[:,1] = y[:,0]
	x[:,2] = x[:,0]
	y[:,2] = y[:,0]+w
	x[:,3] = x[:,1]
	y[:,3] = y[:,2]
	x[:,4] = x[:,2]
	y[:,4] = coords[:,3]
	x[:,5] = x[:,3]
	y[:,5] = y[:,4]
	
	return x.astype('int'),y.astype('int')
	
def generate_3d_coordinates_horizontal(coords):
	(n,quatre) = coords.shape
	x = np.zeros((n,8))
	y = np.zeros((n,8))
	w = np.floor(((coords[:,2]-coords[:,0])+1.)/3.).astype('int')
	
	x[:,0] = coords[:,0]-1
	y[:,0] = coords[:,1]-1
	x[:,1] = x[:,0]+w
	y[:,1] = y[:,0]
	x[:,2] = x[:,1]+w
	y[:,2] = y[:,0]
	x[:,3] = coords[:,2]
	y[:,3] = y[:,0]
	x[:,4] = x[:,0]
	y[:,4] = coords[:,3]
	x[:,5] = x[:,1]
	y[:,5] = y[:,4]
	x[:,6] = x[:,2]
	y[:,6] = y[:,4]
	x[:,7] = x[:,3]
	y[:,7] = y[:,4]
	
	return x.astype('int'),y.astype('int')
	
def generate_3d_coordinates_vertical(coords):
	(n,quatre) = coords.shape
	x = np.zeros((n,8))
	y = np.zeros((n,8))
	w = np.floor(((coords[:,3]-coords[:,1])+1.)/3.).astype('int')
	
	x[:,0] = coords[:,0]-1
	y[:,0] = coords[:,1]-1
	x[:,1] = coords[:,2]
	y[:,1] = y[:,0]
	x[:,2] = x[:,0]
	y[:,2] = y[:,0]+w
	x[:,3] = x[:,1]
	y[:,3] = y[:,2]
	x[:,4] = x[:,2]
	y[:,4] = y[:,2]+w
	x[:,5] = x[:,3]
	y[:,5] = y[:,4]
	x[:,6] = x[:,4]
	y[:,6] = coords[:,3]
	x[:,7] = x[:,5]
	y[:,7] = y[:,6]
	
	return x.astype('int'),y.astype('int')
	
def generate_square_coordinates(coords):
	(n,quatre) = coords.shape
	x = np.zeros((n,9))
	y = np.zeros((n,9))
	
	wx = np.floor(0.5*((coords[:,2]-coords[:,0])+1.)).astype('int')
	wy = np.floor(0.5*((coords[:,3]-coords[:,1])+1.)).astype('int')

	x[:,0] = coords[:,0]-1
	y[:,0] = coords[:,1]-1
	
	x[:,1] = x[:,0]+wx
	y[:,1] = y[:,0]
	
	x[:,2] = coords[:,2]
	y[:,2] = y[:,0]
	
	x[:,3] = x[:,0]
	y[:,3] = y[:,0]+wy
	
	x[:,4] = x[:,1]
	y[:,4] = y[:,3]
	
	x[:,5] = x[:,2]
	y[:,5] = y[:,4]
	
	x[:,6] = x[:,3]
	y[:,6] = coords[:,3]
	
	x[:,7] = x[:,4]
	y[:,7] = y[:,6]
	
	x[:,8] = x[:,5]
	y[:,8] = y[:,6]
	
	return x.astype('int'),y.astype('int')

def pad_integral(intg):
	(h,w) = intg.shape
	nintg = np.zeros((h+1,w+1))
	nintg[1:,1:]=intg
	return nintg
	
def compute_features(intg,x,y,coords_h2,coords_v2,coords_h3,coords_v3,coords_sq):
	
	pad_intg = pad_integral(intg)
	x = x+1
	y = y+1
	(h,w) = pad_intg.shape
	h-=1
	w-=1
	
	(n_h2,quatre) = coords_h2.shape
	(n_v2,quatre) = coords_v2.shape
	(n_h3,quatre) = coords_h3.shape
	(n_v3,quatre) = coords_v3.shape
	(n_sq,quatre) = coords_sq.shape
	
	
	ndata = x.size
	coords = np.zeros((ndata,4))
	dataset = np.zeros((ndata,n_h2+n_v2+n_h3+n_v3+n_sq))
	feature_index = 0
	
	for i in range(n_h2):
		coords[:,0] = (x+coords_h2[i,0])
		coords[:,1] = (y+coords_h2[i,1])
		coords[:,2] = (x+coords_h2[i,2])
		coords[:,3] = (y+coords_h2[i,3])
		(xc,yc) = generate_2d_coordinates_horizontal(coords)
		xc = xc.clip(min=0,max=w)
		yc = yc.clip(min=0,max=h)
		zero   = pad_intg[yc[:,0],xc[:,0]]
		un     = pad_intg[yc[:,1],xc[:,1]]
		deux   = pad_intg[yc[:,2],xc[:,2]]
		trois  = pad_intg[yc[:,3],xc[:,3]]
		quatre = pad_intg[yc[:,4],xc[:,4]]
		cinq   = pad_intg[yc[:,5],xc[:,5]]
		dataset[:,feature_index] = zero+(2*un)+(-deux)+trois+(-2*quatre)+cinq
		feature_index += 1
	
	for i in range(n_v2):
		coords[:,0] = x+coords_v2[i,0]
		coords[:,1] = y+coords_v2[i,1]
		coords[:,2] = x+coords_v2[i,2]
		coords[:,3] = y+coords_v2[i,3]
		(xc,yc) = generate_2d_coordinates_vertical(coords)
		xc = xc.clip(min=0,max=w)
		yc = yc.clip(min=0,max=h)
		zero   = pad_intg[yc[:,0],xc[:,0]]
		un     = pad_intg[yc[:,1],xc[:,1]]
		deux   = pad_intg[yc[:,2],xc[:,2]]
		trois  = pad_intg[yc[:,3],xc[:,3]]
		quatre = pad_intg[yc[:,4],xc[:,4]]
		cinq   = pad_intg[yc[:,5],xc[:,5]]
		dataset[:,feature_index] = zero+(-un)+(-2*deux)+(2*trois)+quatre-cinq
		feature_index+=1
	
	for i in range(n_h3):
		coords[:,0] = x+coords_h3[i,0]
		coords[:,1] = y+coords_h3[i,1]
		coords[:,2] = x+coords_h3[i,2]
		coords[:,3] = y+coords_h3[i,3]
		(xc,yc) = generate_3d_coordinates_horizontal(coords)
		xc = xc.clip(min=0,max=w)
		yc = yc.clip(min=0,max=h)
		zero   = pad_intg[yc[:,0],xc[:,0]]
		un     = pad_intg[yc[:,1],xc[:,1]]
		deux   = pad_intg[yc[:,2],xc[:,2]]
		trois  = pad_intg[yc[:,3],xc[:,3]]
		quatre = pad_intg[yc[:,4],xc[:,4]]
		cinq   = pad_intg[yc[:,5],xc[:,5]]
		six    = pad_intg[yc[:,6],xc[:,6]]
		sept   = pad_intg[yc[:,7],xc[:,7]]
		dataset[:,feature_index] = zero+(-2*un)+(2*deux)+(-trois)+(-quatre)+(2*cinq)+(-2*six)+sept
		feature_index += 1
		
	for i in range(n_v3):
		coords[:,0] = x+coords_v3[i,0]
		coords[:,1] = y+coords_v3[i,1]
		coords[:,2] = x+coords_v3[i,2]
		coords[:,3] = y+coords_v3[i,3]
		(xc,yc) = generate_3d_coordinates_vertical(coords)
		xc = xc.clip(min=0,max=w)
		yc = yc.clip(min=0,max=h)
		zero   = pad_intg[yc[:,0],xc[:,0]]
		un     = pad_intg[yc[:,1],xc[:,1]]
		deux   = pad_intg[yc[:,2],xc[:,2]]
		trois  = pad_intg[yc[:,3],xc[:,3]]
		quatre = pad_intg[yc[:,4],xc[:,4]]
		cinq   = pad_intg[yc[:,5],xc[:,5]]
		six    = pad_intg[yc[:,6],xc[:,6]]
		sept   = pad_intg[yc[:,7],xc[:,7]]
		dataset[:,feature_index] = zero+(-un)+(-2*deux)+(2*trois)+(2*quatre)+(-2*cinq)+(-six)+sept
		feature_index += 1
		
	for i in range(n_sq):
		coords[:,0] = x+coords_sq[i,0]
		coords[:,1] = y+coords_sq[i,1]
		coords[:,2] = x+coords_sq[i,2]
		coords[:,3] = y+coords_sq[i,3]
		(xc,yc) = generate_square_coordinates(coords)
		xc = xc.clip(min=0,max=w)
		yc = yc.clip(min=0,max=h)
		zero   = pad_intg[yc[:,0],xc[:,0]]
		un     = pad_intg[yc[:,1],xc[:,1]]
		deux   = pad_intg[yc[:,2],xc[:,2]]
		trois  = pad_intg[yc[:,3],xc[:,3]]
		quatre = pad_intg[yc[:,4],xc[:,4]]
		cinq   = pad_intg[yc[:,5],xc[:,5]]
		six    = pad_intg[yc[:,6],xc[:,6]]
		sept   = pad_intg[yc[:,7],xc[:,7]]
		huit   = pad_intg[yc[:,8],xc[:,8]]
		dataset[:,feature_index] = zero+(-2*un)+deux+(-2*trois)+(4*quatre)+(-2*cinq)+six+(-2*sept)+huit
		feature_index += 1
	
	return dataset

def build_dataset_image_offset(repository,image_number,xc,yc,dmax,nsamples,h2,v2,h3,v3,sq):
	intg = build_integral_image(readimage(repository,image_number))
	(h,w) = intg.shape
	
	"""
	r = np.arange(-R,R+1)
	xs = []
	ys = []
	rep_x = []
	rep_y = []
	for x in r:
		for y in r:
			if(np.linalg.norm([x,y])<=R):
				xs.append(xc+x)
				ys.append(yc+y)
				rep_x.append(x)
				rep_y.append(y)
	
	n_1 = len(xs)
	n_0 = int(np.round(n_1*P))
	r = R+(np.random.ranf(n_0)*(RMAX-R))
	angle = np.random.ranf(n_0)*2.*np.pi
	
	x0 = xc+np.round(r*np.cos(angle)).astype('int')
	y0 = yc+np.round(r*np.sin(angle)).astype('int')
	
	x0 = x0.clip(min=0,max=w-1)
	y0 = y0.clip(min=0,max=h-1)
	"""
	
	rep_x = np.random.randint(-dmax,dmax+1,nsamples)
	rep_y = np.random.randint(-dmax,dmax+1,nsamples)
	
	xs = xc+rep_x
	ys = yc+rep_y
	
	rep = np.zeros((nsamples,2))
	rep[:,0]=rep_x
	rep[:,1]=rep_y
	return compute_features(intg,xs,ys,h2,v2,h3,v3,sq),rep

def bdio_helper(jobargs):
	return build_dataset_image_offset(*jobargs)
	
def build_dataset_image_offset_mp(repository,xc,yc,ims,dmax,nsamples,h2,v2,h3,v3,sq,n_jobs):
	nimages = xc.size
	jobargs = [(repository,ims[image_number],xc[image_number],yc[image_number],dmax,nsamples,h2,v2,h3,v3,sq) for image_number in range(nimages)]
	P = Pool(n_jobs)
	T = P.map(bdio_helper,jobargs)
	P.close()
	P.join()
	DATASET = None
	REP = None
	IMG = None
	
	b = 0
	
	for i in range(nimages):
		(data,r) = T[i]
		if(i==0):
			(h,w) = data.shape
			DATASET = np.zeros((h*nimages,w))
			REP = np.zeros((h*nimages,2))
			IMG = np.zeros(h*nimages)
		next_b = b+h
		DATASET[b:next_b,:]=data
		REP[b:next_b,:]=r
		IMG[b:next_b]=i
		b = next_b
		
	return DATASET,REP,IMG
	
def build_vote_map(repository,image_number,clf,h2,v2,h3,v3,sq,mx,my,cm,stepc):
	intg = build_integral_image(readimage(repository,image_number))
	(h,w) = intg.shape
	
	vote_map = np.zeros((h,w))
	

	coords = np.array([[x,y] for x in range(0,w,stepc) for y in range(0,h,stepc)]).astype(int)
	
	y_v = coords[:,1]
	x_v = coords[:,0]
	
	step = 50000
	
	b = 0
	
	rep = np.zeros((step,2))
	
	hash_map = {}
	
	while(b<x_v.size):
		b_next = min(b+step,x_v.size)
		offsets = clf.predict(compute_features(intg,x_v[b:b_next],y_v[b:b_next],h2,v2,h3,v3,sq))
		n_trees = len(offsets)
		off_size = int(b_next-b)
		
		offsets = np.array(offsets)
		toffsize = off_size*n_trees
		offsets = offsets.reshape((toffsize,2))
		
		offsets[:,0] = np.tile(x_v[b:b_next],n_trees)-offsets[:,0]
		offsets[:,1] = np.tile(y_v[b:b_next],n_trees)-offsets[:,1]
		
		t, = np.where(offsets[:,0]<0)
		offsets = np.delete(offsets,t,axis=0)
		t, = np.where(offsets[:,1]<0)
		offsets = np.delete(offsets,t,axis=0)
		t, = np.where(offsets[:,0]>=w)
		offsets = np.delete(offsets,t,axis=0)
		t, = np.where(offsets[:,1]>=h)
		offsets = np.delete(offsets,t,axis=0)
		#print offsets.shape,toffsize
		(toffsize,tamere) = offsets.shape
		for i in range(toffsize):
			vote_map[offsets[i,1],offsets[i,0]] += 1

		b=b_next
	
	return vote_map
	
if __name__ == "__main__":
	n_jobs = 4
	repository =    sys.argv[1]
	ip =        int(sys.argv[2])	
	D_MAX =     int(sys.argv[3])#300
	n_samples = int(sys.argv[4])#600
	W =         int(sys.argv[5])#50
	n =         int(sys.argv[6])#50
	T=          int(sys.argv[7])#50
	step =      int(sys.argv[8])#8

	(Xc,Yc) = getcoords(repository.rstrip('/')+'/txt/')
	(nims,nldms) = Xc.shape
	if(nldms==15):
		dset = 'droso'
	elif(nldms==19):
		dset = 'cepha'
	else:
		dset = 'zebra'	
	h2 = generate_2_horizontal(W,n)
	v2 = generate_2_vertical(W,n)
	h3 = generate_3_horizontal(W,n)
	v3 = generate_3_vertical(W,n)
	sq = generate_square(W,n)
	
	(dataset,rep,img) = build_dataset_image_offset_mp(repository,Xc[:,ip],Yc[:,ip],D_MAX,n_samples,h2,v2,h3,v3,sq,n_jobs)
	
	nims = int(np.max(img)+1)
	
	for k in range(10):
		t, = np.where(np.mod(img,10)!=k)
		tr_data = dataset[t,:]
		tr_rep  = rep[t]
		clf = VotingTreeRegressor(n_estimators=T,n_jobs=n_jobs)
		clf = clf.fit(tr_data,tr_rep)
		t, = np.where(np.mod(np.arange(nims),10)==k)
		mx = np.mean(Xc[t,ip])
		my = np.mean(Yc[t,ip])
		
		p = np.zeros((2,t.size))
		p[0,:]=Xc[t,ip]
		p[1,:]=Yc[t,ip]
		
		cm = np.cov(p)
		
		for search_image in range(nims):
			if(np.mod(search_image,10)==k):
				vote_map = build_vote_map(repository,search_image,clf,h2,v2,h3,v3,sq,mx,my,cm,step)
				np.savez_compressed("/home/genmol/tmp/rvandael/%s_vote_map_%d_%d_%d_%d_%d_%d_%d_%d.npy"%(dset,search_image,ip,D_MAX,n_samples,W,n,T,step),vote_map)
