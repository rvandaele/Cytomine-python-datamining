from LonelyTrees import LonelyTrees
from LonelyTreesRegressor import LonelyTreesRegressor
from ldmtools import *
import numpy as np
from multiprocessing import Pool
import scipy.ndimage as snd
import scipy.misc as misc
from scipy.stats import multivariate_normal
from sumproduct import Variable,Factor,FactorGraph
import matplotlib.pyplot as plt
"""
Phase 1 : Pixel filtering
"""

def dataset_from_coordinates(img,x,y,feature_offsets):
	(h,w) = img.shape
	original_values = img[y.clip(min=0,max=h-1),x.clip(min=0,max=w-1)]
	dataset = np.zeros((x.size,feature_offsets[:,0].size))
	for i in range(feature_offsets[:,0].size):
		dataset[:,i] = original_values-img[(y+feature_offsets[i,1]).clip(min=0,max=h-1),(x+feature_offsets[i,0]).clip(min=0,max=w-1)]
	return dataset
	
def image_dataset_phase_1(repository,image_number,x,y,feature_offsets,R_offsets,delta,P):
	#print "PHASE 1 IMAGE NUMBER ",image_number+1
	
	img = makesize(snd.zoom(readimage(repository,image_number),delta),1)
	(h,w) = img.shape
	mask = np.ones((h,w),'bool')
	mask[:,0] = 0
	mask[0,:] = 0
	mask[h-1,:]=0
	mask[:,w-1]=0
	(nroff,blc) = R_offsets.shape
	
	h -= 2
	w -= 2
	x += 1
	y += 1
	
	rep = np.zeros((x.size*nroff)+(P*nroff))
	xs  = np.zeros((x.size*nroff)+(P*nroff)).astype('int')
	ys  = np.zeros((x.size*nroff)+(P*nroff)).astype('int')
	for ip in range(x.size):
		xs[ip*nroff:(ip+1)*nroff] = x[ip]+R_offsets[:,0]
		ys[ip*nroff:(ip+1)*nroff] = y[ip]+R_offsets[:,1]
		rep[ip*nroff:(ip+1)*nroff]=ip
	mask[ys,xs]=0
	(ym,xm) = np.where(mask==1)
	perm = np.random.permutation(ym.size)[0:P*nroff]
	ym = ym[perm]
	xm = xm[perm]
	xs[x.size*nroff:]=xm
	ys[y.size*nroff:]=ym
	rep[x.size*nroff:]=x.size
	
	mask = None	
	dataset = dataset_from_coordinates(img,xs,ys,feature_offsets)
	return dataset,rep
	
def dataset_mp_helper(jobargs):
	return image_dataset_phase_1(*jobargs)
	
def get_dataset_phase_1(repository,n_jobs,feature_offsets,R_offsets,delta,P,k_val):
	p = Pool(n_jobs)
	(Xc,Yc) = getcoords(repository.rstrip('/')+'/txt/')
	(nims,nldms) = Xc.shape
	t, = np.where(np.mod(np.arange(nims),10)!=k_val)
	Xc = np.round(Xc*delta).astype('int')
	Yc = np.round(Yc*delta).astype('int')
	(nims,nldms) = Xc.shape
	jobargs = [(repository,i,Xc[i,:],Yc[i,:],feature_offsets,R_offsets,delta,P) for i in t]
	data = p.map(dataset_mp_helper,jobargs)
	p.close()
	p.join()
	
	(nroff,blc) = R_offsets.shape
	
	DATASET = np.zeros((t.size*(nroff*(nldms+P)),feature_offsets[:,0].size))
	REP = np.zeros(t.size*(nroff*(nldms+P)))
	
	b = 0
	for (d,r) in data:
		(nd,nw) = d.shape
		DATASET[b:b+nd,:]=d
		REP[b:b+nd]=r
		b = b+nd
	DATASET = DATASET[0:b,:]
	REP = REP[0:b]
	return DATASET,REP
	
#Input:  path to images, number of processors and specific phase 1 parameters
#Output: a list of extratrees (or a class?) and the F parameters
def build_phase_1_model(repository,n_jobs=1,NT=32,F=100,R=2,sigma=10,delta=0.25,P=1,k_val=0):

	std_matrix = np.eye(2)*(sigma**2)
	feature_offsets = np.round(np.random.multivariate_normal([0,0],std_matrix,NT*F)).astype('int')
	
	R_offsets = []
	for x1 in range(-R,R+1):
		for x2 in range(-R,R+1):
			if(np.linalg.norm([x1,x2])<=R):
				R_offsets.append([x1,x2])
			
	R_offsets = np.array(R_offsets).astype('int')
	
	(dataset,rep) = get_dataset_phase_1(repository,n_jobs,feature_offsets,R_offsets,delta,P,k_val)	
	
	clf = LonelyTrees(n_estimators=NT,n_jobs=n_jobs)
	clf.fit(dataset,rep)
	return clf,feature_offsets


def probability_map_phase_1(repository,image_number,clf,feature_offsets,delta):
	img = makesize(snd.zoom(readimage(repository,image_number),delta),1)
	(h,w) = img.shape
	ys = []
	xs = []
	
	c = np.arange((h-2)*(w-2))
	ys = 1+np.round(c/(w-2)).astype('int')
	xs = 1+np.mod(c,(w-2))
			
	step = 20000
	b=0
	probability_map = None
	nldms = -1
	
	while(b<xs.size):
		
		next_b = min(b+step,xs.size)
		#print b,next_b
		dataset = dataset_from_coordinates(img,xs[b:next_b],ys[b:next_b],feature_offsets)
		probabilities = clf.predict_proba(dataset)
		
		if(nldms==-1):
			(ns,nldms) = probabilities.shape
			probability_map = np.zeros((h-2,w-2,nldms))
		
		for ip in range(nldms):
			probability_map[ys[b:next_b]-1,xs[b:next_b]-1,ip]=probabilities[:,ip]
		b = next_b
	
	return probability_map
		

"""
Phase 2: Agregation
"""

def image_dataset_phase_2(repository,image_number,x,y,feature_offsets,R_offsets,delta):
	#print "PHASE 2 IMAGE NUMBER",image_number+1
	
	img = makesize(snd.zoom(readimage(repository,image_number),delta),1)
	(h,w) = img.shape
	mask = np.ones((h,w),'bool')
	mask[:,0] = 0
	mask[0,:] = 0
	mask[h-1,:]=0
	mask[:,w-1]=0
	(nroff,blc) = R_offsets.shape
	
	h -= 2
	w -= 2
	x += 1
	y += 1
	
	rep = np.zeros((nroff,2))
	number = image_number
	
	xs = (x+R_offsets[:,0]).astype('int')
	ys = (y+R_offsets[:,1]).astype('int')
	
	rep[:,0]=R_offsets[:,0]
	rep[:,1]=R_offsets[:,1]
	dataset = dataset_from_coordinates(img,xs,ys,feature_offsets)
	return dataset,rep,number

def dataset_mp_helper_phase_2(jobargs):
	return image_dataset_phase_2(*jobargs)
	
def get_dataset_phase_2(repository,tr_images,image_ids,n_jobs,id_term,feature_offsets,R_offsets,delta):
	p = Pool(n_jobs)
	#(Xc,Yc) = getcoords(repository.rstrip('/')+'/txt/')
	(Xc,Yc,Xp,Yp,ims) = getcoords(repository.rstrip('/') + '/txt/',id_term)
	nims = Xc.size
	jobargs = []
	for i in range(nims):
		if(image_ids[i] in tr_images):
			jobargs.append((repository,image_ids[i],Xc[i],Yc[i],feature_offsets,R_offsets,delta))
	data = p.map(dataset_mp_helper_phase_2,jobargs)
	p.close()
	p.join()
	
	(nroff,blc) = R_offsets.shape
	nims = len(tr_images)
	DATASET = np.zeros((nims*nroff,feature_offsets[:,0].size))
	REP = np.zeros((nims*nroff,2))
	NUMBER = np.zeros(nims*nroff)
	
	b = 0
	for (d,r,n) in data:
		(nd,nw) = d.shape
		DATASET[b:b+nd,:]=d
		REP[b:b+nd,:]=r
		NUMBER[b:b+nd]=n
		b = b+nd
	DATASET = DATASET[0:b,:]
	REP = REP[0:b]
	NUMBER = NUMBER[0:b]
	return DATASET,REP,NUMBER
	
def build_phase_2_model(repository,tr_image=[],image_ids=[],n_jobs=1,IP=0,NT=32,F=100,R=3,N=500,sigma=10,delta=0.25):

	std_matrix = np.eye(2)*(sigma**2)
	feature_offsets = np.round(np.random.multivariate_normal([0,0],std_matrix,NT*F)).astype('int')
	
	R_offsets = np.zeros((N,2))
	dis = np.random.ranf(N)*R
	ang = np.random.ranf(N)*2*np.pi
	
	R_offsets[:,0] = np.round((dis*np.cos(ang))).astype('int')
	R_offsets[:,1] = np.round((dis*np.sin(ang))).astype('int')
	
	(dataset,rep,number) = get_dataset_phase_2(repository,tr_image,image_ids,n_jobs,IP,feature_offsets,R_offsets,delta)
	
	return dataset,rep,number,feature_offsets

def filter_perso(img,filter_size):
	offsets = []
	r = range(-filter_size,filter_size+1)
	for r1 in r:
		for r2 in r:
			if(np.linalg.norm([r1,r2])<=filter_size and r1!=0 and r2!=0):
				offsets.append([r1,r2])
	offsets = np.array(offsets)
	
	(h,w) = img.shape
	y,x = np.where(img>0.)
	nimg = np.zeros((h,w))
	for i in range(x.size):
		val = img[y[i],x[i]]
		if(np.sum(val<img[(y[i]+offsets[:,1]).clip(min=0,max=h-1),(x[i]+offsets[:,0]).clip(min=0,max=w-1)])==0):
			nimg[y[i],x[i]]=val

	return nimg
	
def agregation_phase_2(repository,image_number,ip,probability_maps,reg,delta,feature_offsets,filter_size,beta,n_iterations):
	img = makesize(snd.zoom(readimage(repository,image_number),delta),1)
	(h,w,nldms) = probability_maps.shape
	nldms-=1
	mh = h-1
	mw = w-1
	for iteration in range(n_iterations):
		y,x = np.where(probability_maps[:,:,ip]>=beta*np.max(probability_maps[:,:,ip]))
		#if(ip==0):
		#	print "iteration numero %d il y a %d a reperer et %d plus grand que 0"%(iteration+1,y.size,np.sum(probability_maps[:,:,ip]>0))
		dataset = dataset_from_coordinates(img,x+1,y+1,feature_offsets)
		offsets = reg.predict(dataset)
		n_x = (x-offsets[:,0]).clip(min=0,max=mw)
		n_y = (y-offsets[:,1]).clip(min=0,max=mh)
		new_pmap = np.zeros((h,w))
		for i in range(n_x.size):
			new_pmap[n_y[i],n_x[i]] += probability_maps[y[i],x[i],ip]
		probability_maps[:,:,ip]=new_pmap
		probability_maps[0,:,ip]=0
		probability_maps[:,0,ip]=0
		probability_maps[mh,:,ip]=0
		probability_maps[:,mw,ip]=0

	return filter_perso(probability_maps[:,:,ip],filter_size)
	
"""
Phase 3 : Markov Random Field
"""
"""
def build_edgematrix_phase_3(repository,sde,delta,T,x_candidates,y_candidates,k_val):
	(Xc,Yc) = getcoords(repository.rstrip('/')+'/txt/')
	Xc = Xc*delta
	Yc = Yc*delta
	(nims,nldms) = Xc.shape
	
	t, = np.where(np.mod(np.arange(nims),10)!=k_val)
	Xc = Xc[t,:]
	Yc = Yc[t,:]
	(nims,nldms) = Xc.shape
	
	differential_entropy = np.eye(nldms)+np.inf
	
	c1 = np.zeros((nims,2))
	c2 = np.zeros((nims,2))
	
	std_matrix = np.eye(2)*(sde**2)
	
	img = readimage(repository,1)
	(height,width) = img.shape
	height = int(np.round(height*delta))
	width = int(np.round(width*delta))
	img = None
	coords = np.zeros((height*width,2))
	coords[:,0] = np.arange(height*width)/width
	coords[:,1] = np.mod(np.arange(height*width),width)
	
	for ldm1 in range(nldms):
		c1[:,0] = Xc[:,ldm1]
		c1[:,1] = Yc[:,ldm1]
		for ldm2 in range(ldm1+1,nldms):
			c2[:,0] = Xc[:,ldm2]
			c2[:,1] = Yc[:,ldm2]
			
			diff = c1-c2
			gaussians = [multivariate_normal(diff[i,:],std_matrix) for i in range(nims)]
			
			s = 0
			for i in range(nims):
				s = s+gaussians[i].pdf(coords)
			s = s/float(nims)
			s = np.log(s)
			s = -np.sum(s)
			
			#print "pair %d %d %f"%(ldm1,ldm2,s)
			
			differential_entropy[ldm1,ldm2] = s
			differential_entropy[ldm2,ldm1] = s
	
	edges = np.zeros((nldms,T))
	
	for ldm in range(nldms):
		edges[ldm,:] = np.argsort(differential_entropy[ldm,:])[0:T]
	
	(nldms,ncandidates) = x_candidates.shape
	
	B_mat = np.zeros((ncandidates,ncandidates,T*nldms))
	
	c = 0
	for ip in range(nldms):
		c1[:,0] = Xc[:,ip]
		c1[:,1] = Yc[:,ip]
		for ipl in edges[ip,:]:
			c2[:,0] = Xc[:,ipl]
			c2[:,1] = Yc[:,ipl]
			
			diff = c1-c2
			
			gaussians = [multivariate_normal(diff[i,:],std_matrix) for i in range(nims)]
			
			for cand1 in range(ncandidates):
				pos1 = np.array([x_candidates[ip,cand1],y_candidates[ip,cand1]])
				for cand2 in range(ncandidates):
					pos2 = np.array([x_candidates[ipl,cand2],y_candidates[ipl,cand2]])
					diff = pos1-pos2
					B_mat[cand1,cand2,c] = np.max([gaussians[i].pdf(diff) for i in range(nims)])
			c+=1
			
	B_mat = B_mat/multivariate_normal([0,0],std_matrix).pdf([0,0])	
	for ip in range(nldms):
		for i in range(ncandidates):
			B_mat[i,:,ip] = B_mat[i,:,ip]/np.sum(B_mat[i,:,ip])
			
	return edges,B_mat
"""
"""
def evaluate_solution(probability_map_phase_2,b_mat,edges,cands,x_cand,y_cand):
		return probability_map_phase_2[]
"""	
def compute_final_solution_phase_3(repository,probability_map_phase_2,ncandidates,sde,delta,T,k_val):
	(height,width,nldms) = probability_map_phase_2.shape
	nldms-=1
	
	x_candidates = np.zeros((nldms,ncandidates))
	y_candidates = np.zeros((nldms,ncandidates))
	
	
	for i in range(nldms):
		val = np.sort(-probability_map_phase_2[:,:,i].flatten())[ncandidates]
		(y,x) = np.where(probability_map_phase_2[:,:,i]>=val)
		
		if(y.size>ncandidates):
			vals = -probability_map_phase_2[y,x,i]
			order = np.argsort(vals)[0:ncandidates]
			y = y[order]
			x = x[order]
			
		x_candidates[i,:] = x
		y_candidates[i,:] = y
	
	(edges,b_mat) = build_edgematrix_phase_3(repository,sde,delta,T,x_candidates,y_candidates,k_val)
	(hh,ww,dd) = b_mat.shape
	
	g = FactorGraph(silent=True)
	nodes = [Variable('x%d'%i,ncandidates) for i in range(nldms)]
	#print b_mat.shape,ncandidates
	for i in range(dd):
		x1 = int(np.floor(float(i)/float(T)))
		x2 = int(edges[x1,int(np.mod(i,T))])
		g.add(Factor('f2_%d'%i,b_mat[:,:,i]))
		g.append('f2_%d'%i,nodes[x1])
		g.append('f2_%d'%i,nodes[x2])
		
	for i in range(nldms):
		g.add(Factor('f1_%d'%i,probability_map_phase_2[y_candidates[i,:].astype('int'),x_candidates[i,:].astype('int')]/np.sum(probability_map_phase_2[y_candidates[i,:].astype('int'),x_candidates[i,:].astype('int')])))
		g.append('f1_%d'%i,nodes[i])

	g.compute_marginals()
	
	x_final = np.zeros(nldms)
	y_final = np.zeros(nldms)
	
	for i in range(nldms):
		amin = np.argmax(g.nodes['x%d'%i].marginal())
		x_final[i] = x_candidates[i,amin]
		y_final[i] = y_candidates[i,amin]
	return x_final/delta,y_final/delta


if __name__ == "__main__":
	
	#PATH TO DATASET
	repository = sys.argv[1]#'/home/remy/datasets/bigres/'
	(Xc,Yc) = getcoords(repository.rstrip('/')+'/txt/')
	(nims,nldms) = Xc.shape

	#COMPUTER SPECIFIC PARAMETERS
	n_jobs = 4
	#PHASE 1 PARAMETERS
	
	#example on zephyros: python donner_validation.py /home/remy/datasets/all_dros/trainbmp/ 32 100 2 10 0.25 1 3 32 100 3 0.5 3 3 50 4
	#example on zephyros: print "python donner_validation.py %s %d %d %d %f %f %d %d %d %d %d %f %d %d %f %d"%(NT_P1,F_P1,R_P1,sigma,delta,P,R_P2,NT_P2,F_P2,filter_size,beta,n_iterations,ncandidates,sde,T)
	
	if(nldms==15):
		datatype='droso'
	elif(nldms==19):
		datatype='cepha'
	else:
		datatype='zebra'
	#UN LANDMARK A LA FOIS
	ip = int(sys.argv[2])
	#PARAMETRE PHASE 1
	NT_P1 = int(sys.argv[3])#32 tree number for classification
	F_P1  = int(sys.argv[4])#100 number of features
	R_P1  = int(sys.argv[5])#2 radius in which landmark samples are taken (they are ALL taken so it can increase memory)
	sigma = float(sys.argv[6])#10 gaussian variance for the feature (the bigger, the farher the features can possibly be)
	delta = float(sys.argv[7])#0.25 resizing the image (between ]0,1], the new images will be of size delta*height x delta*width
	P     = int(sys.argv[8])#1 proportion of background offsets sampled compared to a SINGLE landmark
	#PHASE 2 PARAMETERS
	R_P2 = int(sys.argv[9])#3 radius in which regression samples are taken
	ns_P2 = int(sys.argv[10]) # number of samples taken in the radius R_P2
	NT_P2 = int(sys.argv[11])#32 number of regression trees for EACH of the #LANDMARKS forests
	F_P2 = int(sys.argv[12])#100 number of features
	filter_size = int(sys.argv[13]) #3 only sample the bigger agregated probability in a radius filter_size
	beta = float(sys.argv[14])#0.5 only consider pixels with probabilities >= beta*max(prob_img)
	n_iterations = int(sys.argv[15])#3 number of iterations during the agregation phase
	
	errors = np.zeros((nims,nldms))
	#sys.exit()
	(dataset,rep,number,feature_offsets_2) = build_phase_2_model(repository,n_jobs=n_jobs,IP=ip,NT=NT_P2,F=F_P2,R=R_P2,N=ns_P2,sigma=sigma,delta=delta)
	
	for K in range(10):
		print "BUILDING DE LA PHASE 2"
		
		reg = LonelyTreesRegressor(n_estimators=NT_P2,n_jobs=n_jobs)
		t, = np.where(np.mod(number,10)!=K)
		reg.fit(dataset[t,:],rep[t,:])
		for i in range(nims):
			if(np.mod(i,10)==K):
				probability_map = np.load('/home/genmol/tmp/rvandael/donner/phase1/%s-phase1-%d-%d-%d-%d-%3.3f-%3.3f-%d.npy.npz'%(datatype,i,NT_P1,F_P1,R_P1,sigma,delta,P))['arr_0']#np.load('/home/genmol/tmp/rvandael/donner/%s-phase1-%d-%d-%d-%d-%3.3f-%3.3f-%d.npy.npz'%(datatype,i,NT_P1,F_P1,R_P1,sigma,delta,P))
				probability_map_phase_2 = agregation_phase_2(repository,i,ip,probability_map,reg,delta,feature_offsets_2,filter_size,beta,n_iterations)
				np.savez_compressed('/home/genmol/tmp/rvandael/donner/phase2/%s-phase2-%d-%d-%d-%d-%d-%3.3f-%3.3f-%d-%d-%d-%d-%d-%d-%3.3f-%d.npy'%(datatype,i,ip,NT_P1,F_P1,R_P1,sigma,delta,P,R_P2,ns_P2,NT_P2,F_P2,filter_size,beta,n_iterations),probability_map_phase_2)
