from LonelyTrees import LonelyTrees
from LonelyTreesRegressor import LonelyTreesRegressor
from ldmtools import *
import numpy as np
from multiprocessing import Pool
import scipy.ndimage as snd
import scipy.misc as misc
from scipy.stats import multivariate_normal
from sumproduct import Variable,Factor,FactorGraph

"""
Phase 3 : Markov Random Field
"""

def build_edgematrix_phase_3(Xc,Yc,sde,delta,T):
	Xc = Xc*delta
	Yc = Yc*delta
	(nims,nldms) = Xc.shape
	
	(nims,nldms) = Xc.shape
	
	differential_entropy = np.eye(nldms)+np.inf
	
	c1 = np.zeros((nims,2))
	c2 = np.zeros((nims,2))
	
	#std_matrix = np.eye(2)*(sde**2)
	"""
	img = readimage(repository,1)
	(height,width) = img.shape
	height = int(np.round(height*delta))
	width = int(np.round(width*delta))
	img = None
	coords = np.zeros((height*width,2))
	coords[:,0] = np.floor(np.arange(height*width)/width)
	coords[:,1] = np.mod(np.arange(height*width),width)
	"""
	for ldm1 in range(nldms):
		c1[:,0] = Xc[:,ldm1]
		c1[:,1] = Yc[:,ldm1]
		for ldm2 in range(ldm1+1,nldms):
			c2[:,0] = Xc[:,ldm2]
			c2[:,1] = Yc[:,ldm2]
			
			diff = c1-c2
		
			d = diff-np.mean(diff,axis=0)
			d = np.mean(np.sqrt((d[:,0]**2)+(d[:,1]**2)))
			differential_entropy[ldm1,ldm2] = d
			differential_entropy[ldm2,ldm1] = d
	
	edges = np.zeros((nldms,T))
	
	for ldm in range(nldms):
		edges[ldm,:] = np.argsort(differential_entropy[ldm,:])[0:T]
	
	return edges.astype(int)
	
def build_bmat_phase_3(xc,yc,T,x_candidates,y_candidates,edges,sde):

	B_mat = {}#np.zeros((ncandidates,ncandidates,T*nldms))
	
	c = 0
	(nims,nldms) = xc.shape
	c1 = np.zeros((nims,2))
	c2 = np.zeros((nims,2))
	
	std_matrix = np.eye(2)*(sde**2)
	
	for ip in range(nldms):
		c1[:,0] = xc[:,ip]
		c1[:,1] = yc[:,ip]
		for ipl in edges[ip,:]:
			rel = np.zeros((len(x_candidates[ip]),len(x_candidates[ipl])))
			
			c2[:,0] = xc[:,ipl]
			c2[:,1] = yc[:,ipl]
			
			diff = c1-c2
			
			gaussians = [multivariate_normal(diff[i,:],std_matrix) for i in range(nims)]
			
			for cand1 in range(len(x_candidates[ip])):
				pos1 = np.array([x_candidates[ip][cand1],y_candidates[ip][cand1]])
				for cand2 in range(len(x_candidates[ipl])):
					pos2 = np.array([x_candidates[ipl][cand2],y_candidates[ipl][cand2]])
					diff = pos1-pos2
					rel[cand1,cand2] = np.max([gaussians[i].pdf(diff) for i in range(nims)])
			B_mat[(ip,ipl)] = rel/multivariate_normal([0,0],std_matrix).pdf([0,0])
			
	for (ip,ipl) in B_mat.keys():
		rel = B_mat[(ip,ipl)]
		for i in range(len(x_candidates[ip])):
			rel[i,:] = rel[i,:]/np.sum(rel[i,:])
		B_mat[(ip,ipl)]=rel	
	return B_mat
"""
def evaluate_solution(probability_map_phase_2,b_mat,edges,cands,x_cand,y_cand):
		return probability_map_phase_2[]
"""	
def compute_final_solution_phase_3(xc,yc,probability_map_phase_2,ncandidates,sde,delta,T,edges,k_val):
	(height,width,nldms) = probability_map_phase_2.shape
	#nldms-=1
	x_candidates = []#np.zeros((nldms,ncandidates))
	y_candidates = []#np.zeros((nldms,ncandidates))
	
	
	for i in range(nldms):
		val = -np.sort(-probability_map_phase_2[:,:,i].flatten())[ncandidates]
		if(val>0):
			(y,x) = np.where(probability_map_phase_2[:,:,i]>=val)
		else:
			(y,x) = np.where(probability_map_phase_2[:,:,i]>val)

		if(y.size>ncandidates):
			vals = -probability_map_phase_2[y,x,i]
			order = np.argsort(vals)[0:ncandidates]
			y = y[order]
			x = x[order]

		x_candidates.append(x.tolist())
		y_candidates.append(y.tolist())
	
	b_mat = build_bmat_phase_3(xc,yc,T,x_candidates,y_candidates,edges,sde)
	
	#(hh,ww,dd) = b_mat.shape
	
	g = FactorGraph(silent=True)
	nodes = [Variable('x%d'%i,len(x_candidates[i])) for i in range(nldms)]
	#print b_mat.shape,ncandidates
	for ip in range(nldms):
		for ipl in edges[ip,:].astype(int):
			if(ip==0):
				print b_mat[(ip,ipl)].shape
			g.add(Factor('f2_%d_%d'%(ip,ipl),b_mat[(ip,ipl)]))
			g.append('f2_%d_%d'%(ip,ipl),nodes[ip])
			g.append('f2_%d_%d'%(ip,ipl),nodes[ipl])
		
	#ycand = np.array(y_candidates).astype(int)
	#xcand = np.array(x_candidates).astype(int)
	for i in range(nldms):
	
		v = probability_map_phase_2[np.array(y_candidates[i]),np.array(x_candidates[i]),i]
		if(i==0):
			print v.size
		g.add(Factor('f1_%d'%i,v/np.sum(v)))
		g.append('f1_%d'%i,nodes[i])

	g.compute_marginals()
	
	x_final = np.zeros(nldms)
	y_final = np.zeros(nldms)
	
	for i in range(nldms):
		amin = np.argmax(g.nodes['x%d'%i].marginal())
		x_final[i] = x_candidates[i][amin]
		y_final[i] = y_candidates[i][amin]
	return x_final/delta,y_final/delta


if __name__ == "__main__":
	
	#PATH TO DATASET
	repository = sys.argv[1]#'/home/remy/datasets/bigres/'
	(Xc,Yc) = getcoords(repository.rstrip('/')+'/txt/')
	(nims,nldms) = Xc.shape
	if(nldms==15):
		datatype='droso'
	elif(nldms==19):
		datatype='cepha'
	else:
		datatype='zebra'
	#COMPUTER SPECIFIC PARAMETERS
	n_jobs = 4
	#PHASE 1 PARAMETERS
	
	#example on zephyros: python donner_validation.py /home/remy/datasets/all_dros/trainbmp/ 32 100 2 10 0.25 1 3 32 100 3 0.5 3 3 50 4
	#example on zephyros: print "python donner_validation.py %s %d %d %d %f %f %d %d %d %d %d %f %d %d %f %d"%(NT_P1,F_P1,R_P1,sigma,delta,P,R_P2,NT_P2,F_P2,filter_size,beta,n_iterations,ncandidates,sde,T)
	
	NT_P1 = int(sys.argv[2])#32 tree number for classification
	F_P1  = int(sys.argv[3])#100 number of features
	R_P1  = int(sys.argv[4])#2 radius in which landmark samples are taken (they are ALL taken so it can increase memory)
	sigma = float(sys.argv[5])#10 gaussian variance for the feature (the bigger, the farher the features can possibly be)
	delta = float(sys.argv[6])#0.25 resizing the image (between ]0,1], the new images will be of size delta*height x delta*width
	P     = int(sys.argv[7])#1 proportion of background offsets sampled compared to a SINGLE landmark
	
	#PHASE 2 PARAMETERS
	R_P2 = int(sys.argv[8])#3 radius in which regression samples are taken
	ns_P2 = int(sys.argv[9]) # number of samples taken in the radius R_P2
	NT_P2 = int(sys.argv[10])#32 number of regression trees for EACH of the #LANDMARKS forests
	F_P2 = int(sys.argv[11])#100 number of features
	filter_size = int(sys.argv[12]) #3 only sample the bigger agregated probability in a radius filter_size
	beta = float(sys.argv[13])#0.5 only consider pixels with probabilities >= beta*max(prob_img)
	n_iterations = int(sys.argv[14])#3 number of iterations during the agregation phase
	
	#PHASE 3 PARAMETERS
	ncandidates = int(sys.argv[15])#3 number of candidates to consider for each landmark
	sde = float(sys.argv[16])#50. variance value for the gaussian functions
	T = int(sys.argv[17])#4 number of edges for each landmark
	
	errors = np.zeros((nims,nldms))
	for K in range(10):
		edges,xc,yc = build_edgematrix_phase_3(Xc,Yc,sde,delta,T,K)
		for i in range(nims):
			if(np.mod(i,10)==K):
				
				probability_map_phase_2 = np.load('/home/genmol/tmp/rvandael/donner/phase2/%s-phase2-%d-%d-%d-%d-%d-%3.3f-%3.3f-%d-%d-%d-%d-%d-%d-%3.3f-%d.npy.npz'%(datatype,i,0,NT_P1,F_P1,R_P1,sigma,delta,P,R_P2,ns_P2,NT_P2,F_P2,filter_size,beta,n_iterations))['arr_0']
				(h,w) = probability_map_phase_2.shape
				pmptot = np.zeros((h,w,nldms))
				pmptot[:,:,0] = probability_map_phase_2
				for ip in range(1,nldms):
					pmptot[:,:,ip] = probability_map_phase_2 = np.load('/home/genmol/tmp/rvandael/donner/phase2/%s-phase2-%d-%d-%d-%d-%d-%3.3f-%3.3f-%d-%d-%d-%d-%d-%d-%3.3f-%d.npy.npz'%(datatype,i,ip,NT_P1,F_P1,R_P1,sigma,delta,P,R_P2,ns_P2,NT_P2,F_P2,filter_size,beta,n_iterations))['arr_0']
				
				#pmptot = np.random.ranf((900,1440,15))
				
				
				x_final,y_final = compute_final_solution_phase_3(xc,yc,pmptot,ncandidates,sde,delta,T,edges,k_val=K)
				for ip in range(nldms):
					errors[i,ip] = np.linalg.norm([Xc[i,ip]-x_final[ip],Yc[i,ip]-y_final[ip]])
					print "IMAGE %d LANDMARK %d DETECTED AT %f %f SHOULD BE AT %f %f ERROR %f"%(i+1,ip,x_final[ip],y_final[ip],Xc[i,ip],Yc[i,ip],errors[i,ip])

	for i in range(nldms):
		print "MEAN LANDMARK %d ERROR %f"%(i,np.mean(errors[i,:]))
	print "GLOBAL ERROR %f"%np.mean(errors)
