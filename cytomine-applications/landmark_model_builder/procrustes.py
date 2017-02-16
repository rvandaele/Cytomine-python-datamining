import numpy as np
from ldmtools import *
from scipy.sparse.linalg import eigs
import sys

"""
Le but de ce script est, en recevant une matrice de coordonnees, de toutes les
aligner avant de pouvoir lancer une PCA
"""

"""
Les donnees sont presentees lignes par ligne
"""
def procrustes(x_coords,y_coords):

	(ndata,nldms) = x_coords.shape
	
	#1. Centrer les formes en 0,0
	mean_x = np.mean(x_coords,axis=1)
	mean_y = np.mean(y_coords,axis=1)
	coords_centered = np.zeros((ndata,2*nldms))
	for i in range(nldms):
		coords_centered[:,i] = x_coords[:,i]-mean_x
		coords_centered[:,i+nldms] = y_coords[:,i]-mean_y
	
	#2. Fixer une forme t.q sa norme soit 1, je prends la premiere
	coords_centered[0,:] = coords_centered[0,:]/np.linalg.norm(coords_centered[0,:])
	
	#3. Scaler et rotater les autres
	
	c = np.zeros((2,nldms))
	for i in range(1,ndata):
		a = np.dot(coords_centered[i,:],coords_centered[0,:])/(np.linalg.norm(coords_centered[i,:])**2)
		b = np.sum((coords_centered[i,0:nldms]*coords_centered[0,nldms:])-(coords_centered[0,:nldms]*coords_centered[i,nldms:]))/(np.linalg.norm(coords_centered[i,:])**2)
		s = np.sqrt((a**2)+(b**2))
		theta = np.arctan(b/a)
		
		scaling_matrix = s*np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
		c[0,:]=coords_centered[i,0:nldms]
		c[1,:]=coords_centered[i,nldms:]
		
		new_c = np.dot(scaling_matrix,c)
		
		coords_centered[i,0:nldms]=new_c[0,:]
		coords_centered[i,nldms:]=new_c[1,:]
	
	return coords_centered
	#print np.max(np.mean(x_coords_centered,axis=1)),np.max(np.mean(y_coords_centered,axis=1))
	
"""
Une ligne = une donnee
"""
def apply_pca(coords,k):
	(ndata,nldms) = coords.shape
	m = np.mean(coords,axis=0).reshape((nldms,1))
	
	mat = np.zeros((nldms,nldms))
	
	for i in range(ndata):
		v = coords[i,:].reshape((nldms,1))
		d = v-m
		mat = mat+np.dot(d,d.T)

	mat = mat/float(nldms-1)

	#vectors[:,i] is the eigenvector corresponding to the eigenvalue values[i], already sorted in descending order
	(values,vectors) = np.linalg.eig(mat)
	
	return m,vectors[:,0:k]
	
"""
y est centre
"""
def fit_shape(mu,P,ty):
	y = np.copy(ty)
	
	
	(nldms,k) = P.shape
	b = np.zeros((k,1))
	nldm = nldms/2
	c = np.zeros((2,nldm))
	new_y = np.zeros(nldms)

	m_1 = np.mean(y[:nldm])
	m_2 = np.mean(y[nldm:])
	
	y[:nldm] = y[:nldm]-m_1
	y[nldm:] = y[nldm:]-m_2

	ite = 0
	
	while(ite<100):
		
		x = mu+np.dot(P,b)
		#print x[0]
		#Procrustes entre x et y, matcher y sur x
		n2=np.linalg.norm(y)**2
		a=(np.dot(y,x)/n2)[0]
		b=np.sum((y[:nldm]*x[nldm:])-(x[:nldm]*y[nldm:]))/n2
		s=np.sqrt((a**2)+(b**2))
		theta=np.arctan(b/a)
		scaling_matrix = s*np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
		c[0,:]=y[:nldm]
		c[1,:]=y[nldm:]
		
		#sys.exit()
		new_c=np.dot(scaling_matrix,c)
		
		new_y[:nldm]=new_c[0,:]
		new_y[nldm:]=new_c[1,:]
		
		b = np.dot(P.T,new_y.reshape((nldms,1))-mu)
		#y = new_y
		ite += 1
	
	s = 1./s
	theta = -theta
	scaling_matrix = s*np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
	#print s,theta
	c[0,:]=x[:nldm].reshape(nldm)
	c[1,:]=x[nldm:].reshape(nldm)
	new_c=np.dot(scaling_matrix,c)
	new_y[:nldm]=new_c[0,:]+m_1
	new_y[nldm:]=new_c[1,:]+m_2
	return new_y	

def find_best_positions(vote_map,coords,R):
	(h,w,nldms) = vote_map.shape
	
	cs = np.zeros(2*nldms)
	for ip in range(nldms):
		
		x_begin = min(w-1,max(0,coords[ip]-R))
		x_end = max(0,min(coords[ip]+R+1,w-1))
		
		y_begin = min(h-1,max(0,coords[ip+nldms]-R))
		y_end = max(0,min(h-1,coords[ip+nldms]+R+1))
		
		if(x_begin!=x_end and y_begin!=y_end):
			window = vote_map[y_begin:y_end,x_begin:x_end,ip]
			(y,x) = np.where(window==np.max(window))
			cs[ip] = x[0]+x_begin
			cs[ip+nldms] = y[0]+y_begin
		elif(x_begin==x_end and y_begin!=y_end):
			window = vote_map[y_begin:y_end,x_begin,ip]
			y, = np.where(window==np.max(window))
			cs[ip] = x_begin
			cs[ip+nldms] = y[0]+y_begin
		elif(y_begin==y_end and x_begin!=x_end):
			window = vote_map[y_begin,x_begin:x_end,ip]
			x, = np.where(window==np.max(window))
			cs[ip+nldms] = y_begin
			cs[ip] = x[0]+x_begin
		else:
			cs[ip] = x_begin
			cs[ip+nldms] = y_begin
			
	return cs
		
if __name__ == "__main__":

	repository = sys.argv[1].rstrip('/')+'/'
	n_reduc = int(sys.argv[2])#10
	R_max = int(sys.argv[3])#200
	R_min = int(sys.argv[4])#2
	alpha = float(sys.argv[5])#0.5
	(Xc,Yc) = getcoords(repository+'txt/')
	
	D_MAX = int(sys.argv[6])#300
	n_samples = int(sys.argv[7])#600
	W = int(sys.argv[8])#50
	n = int(sys.argv[9])#50
	T = int(sys.argv[10])#50
	step = int(sys.argv[11])#8
	
	(nimages,nldms) = Xc.shape
	if(nldms==15):
		dset='droso'
	elif(nldms==19):
		dset='cepha'
	else:
		dset='zebra'
	ers = np.zeros((nimages,nldms))
	coords = np.zeros(2*nldms)
	for k in range(10):
		t, = np.where(np.mod(np.arange(nimages),10)!=k)
		mx = np.round(np.mean(Xc[t,:],axis=0)).astype(int)
		my = np.round(np.mean(Yc[t,:],axis=0)).astype(int)
		imrange, = np.where(np.mod(np.arange(nimages),10)==k)
		coords[:nldms]=mx
		coords[nldms:]=my
		x_c = procrustes(Xc[t,:],Yc[t,:])
		(mu,P) = apply_pca(x_c,n_reduc)
			
		for i in imrange.astype(int):
			img = np.load('/home/genmol/tmp/rvandael/%s_vote_map_%d_%d_%d_%d_%d_%d_%d_%d.npy.npz'%(dset,i,0,D_MAX,n_samples,W,n,T,step))['arr_0']
			
			(h,w) = img.shape
			vote_maps = np.zeros((h,w,nldms))
			vote_maps[:,:,0] = img
			for ip in range(1,nldms):
				vote_maps[:,:,ip] = np.load('/home/genmol/tmp/rvandael/%s_vote_map_%d_%d_%d_%d_%d_%d_%d_%d.npy.npz'%(dset,i,ip,D_MAX,n_samples,W,n,T,step))['arr_0']
			current_R = R_max
			while(current_R>=R_min):
				coords = np.round(find_best_positions(vote_maps,coords,int(np.round(current_R)))).astype(int)
				coords = np.round(fit_shape(mu,P,coords)).astype(int)
				current_R = current_R*alpha
			for j in range(nldms):
				er = np.linalg.norm([Xc[i,j]-coords[j],Yc[i,j]-coords[j+nldms]])
				print "IMAGE %d LANDMARK %d FOUND IN %f %f BUT REALLY IN %f %f ERROR %f"%(i,j,coords[j],coords[j+nldms],Xc[i,j],Yc[i,j],er)
				ers[i,j] = er

	print np.mean(ers)
		
