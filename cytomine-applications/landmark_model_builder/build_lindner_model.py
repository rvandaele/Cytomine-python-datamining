from LonelyTrees import LonelyTrees
from LonelyTreesRegressor import LonelyTreesRegressor
from ldmtools import *
import numpy as np
from multiprocessing import Pool
import scipy.ndimage as snd
import scipy.misc as misc
from scipy.stats import multivariate_normal
from sumproduct import Variable, Factor, FactorGraph
from sklearn.externals import joblib
from lindner_map import *
from procrustes import *

if __name__ == "__main__":
	"""
	ip =        int(sys.argv[2])
	D_MAX =     int(sys.argv[3])#300
	n_samples = int(sys.argv[4])#600
	W =         int(sys.argv[5])#50
	n =         int(sys.argv[6])#50
	T=          int(sys.argv[7])#50
	step =      int(sys.argv[8])#8

	n_reduc = int(sys.argv[2])#10
	R_max = int(sys.argv[3])#200
	R_min = int(sys.argv[4])#2
	alpha = float(sys.argv[5])#0.5
	"""

	parameters = {
		'cytomine_host': '',
		'cytomine_public_key': '',
		'cytomine_private_key': '',
		'cytomine_id_software': 0,
		'cytomine_base_path': '',
		'cytomine_working_path': '',
		'cytomine_id_project': None,
		'cytomine_id_terms':None,
		'image_type': '',
		'model_njobs': None,
		'model_D_MAX': None,
		'model_n_samples': None,
		'model_W': None,
		'model_n': None,
		'model_T': None,
		'model_step': None,
		'model_n_reduc': None,
		'model_R_MAX': None,
		'model_R_MIN': None,
		'model_alpha': None,
		'model_name': '',
		'model_save_to': '',
		'verbose': False
	}

	p = optparse.OptionParser(description='Cytomine Landmark Detection : Model building', prog='Cytomine Landmark Detector : Model builder', version='0.1')
	p.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default='XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key', type="string", default='YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--cytomine_training_images', type='string', default='all', dest='cytomine_training_images', help="IDs of the training images. Must be separated by commas, no spaces. 'all' takes all the available annotated images.")
	p.add_option('--cytomine_id_terms', type='string', default=1, dest='cytomine_id_terms', help="The identifiers of the terms to create detection models for. Terms must be separated by commas (no spaces). If 'all' is mentioned instead, every terms will be detected.")
	p.add_option('--image_type', type='string', default='jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--model_njobs', type='int', default=4, dest='model_njobs', help="The number of processors used for model building")
	p.add_option('--model_D_MAX', type='int', default=6, dest='model_D_MAX', help="D_MAX parameter.")
	p.add_option('--model_n_samples', type='int', default=200, dest='model_n_samples', help="Number of samples for phase 1.")
	p.add_option('--model_W', type='int', default=100, dest='model_W', help="Window size for feature extraction.")
	p.add_option('--model_n', type='int', default=20, dest='model_n',help="Number of samples extracted.")
	p.add_option('--model_T', type='int', default=50, dest='model_T', help="Number of trees for phase 1.")
	p.add_option('--model_step', type='int', default=3, dest='model_step', help="Step for prediction for phase 1.")
	p.add_option('--model_n_reduc', type='int', default=2, dest='model_n_reduc', help="Size for PCA reduction in phase 2.")
	p.add_option('--model_R_MAX', type='int', default=20, dest='model_R_MAX', help="Maximal radius for phase 2.")
	p.add_option('--model_R_MIN', type='int', default=3, dest='model_R_MIN', help="Minimal radius for phase 2.")
	p.add_option('--model_alpha', type='float', default=0.5, dest='model_alpha', help="Radius reduction parameter for phase 2.")
	p.add_option('--model_save_to', type='string', default='/tmp/', dest='model_save_to', help="Destination for model storage")
	p.add_option('--model_name', type='string', dest='model_name', help="Name of the model (used for saving)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

	options, arguments = p.parse_args(args=sys.argv)

	parameters['cytomine_host'] = options.cytomine_host
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path
	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['cytomine_id_terms'] = options.cytomine_id_terms
	parameters['cytomine_training_images'] = options.cytomine_training_images
	parameters['image_type'] = options.image_type
	parameters['model_njobs'] = options.model_njobs
	parameters['model_D_MAX'] = options.model_D_MAX
	parameters['model_n_samples'] = options.model_n_samples
	parameters['model_W'] = options.model_W
	parameters['model_n'] = options.model_n
	parameters['model_T'] = options.model_T
	parameters['model_step'] = options.model_step
	parameters['model_n_reduc'] = options.model_n_reduc
	parameters['model_R_MAX'] = options.model_R_MAX
	parameters['model_R_MIN'] = options.model_R_MIN
	parameters['model_alpha'] = options.model_alpha
	parameters['model_save_to'] = options.model_save_to
	parameters['model_name'] = options.model_name
	parameters['verbose'] = str2bool(options.verbose)

	if (not parameters['cytomine_working_path'].endswith('/')):
		parameters['cytomine_working_path'] = parameters['cytomine_working_path'] + '/'

	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'], parameters['cytomine_public_key'], parameters['cytomine_private_key'], base_path=parameters['cytomine_base_path'], working_path=parameters['cytomine_working_path'], verbose=parameters['verbose'])

	current_user = cytomine_connection.get_current_user()
	run_by_user_job = False
	if current_user.algo == False:
		user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
	else:
		user_job = current_user

	run_by_user_job = True

	job = cytomine_connection.get_job(user_job.job)
	job = cytomine_connection.update_job_status(job, status=job.RUNNING, progress=0, status_comment="Bulding model...")
	job_parameters = {}
	job_parameters['cytomine_id_terms'] = parameters['cytomine_id_terms']
	job_parameters['model_njobs'] = parameters['model_njobs']
	job_parameters['model_D_MAX'] = parameters['model_D_MAX']
	job_parameters['model_n_samples'] = parameters['model_n_samples']
	job_parameters['model_W'] = parameters['model_W']
	job_parameters['model_n'] = parameters['model_n']
	job_parameters['model_T'] = parameters['model_T']
	job_parameters['model_step'] = parameters['model_step']
	job_parameters['model_n_reduc'] = parameters['model_n_reduc']
	job_parameters['model_R_MAX'] = parameters['model_R_MAX']
	job_parameters['model_R_MIN'] = parameters['model_R_MIN']
	job_parameters['model_alpha'] = parameters['model_alpha']
	model_repo = parameters['model_save_to']

	if (not os.path.isdir(model_repo)):
		os.mkdir(model_repo)

	if run_by_user_job == False:
		job_parameters_values = cytomine_connection.add_job_parameters(user_job.job, cytomine_connection.get_software(parameters['cytomine_id_software']), job_parameters)

	download_images(cytomine_connection, parameters['cytomine_id_project'])
	download_annotations(cytomine_connection, parameters['cytomine_id_project'], parameters['cytomine_working_path'])

	repository = parameters['cytomine_working_path'] + str(parameters['cytomine_id_project']) + '/'
	txt_repository = parameters['cytomine_working_path'] + '%d/txt/' % parameters['cytomine_id_project']

	(xc, yc, xr, yr, ims, term_to_i, i_to_term) = getallcoords(repository.rstrip('/') + '/txt/')
	(nims, nldms) = xc.shape

	if (parameters['cytomine_id_terms'] != 'all'):
		term_list = [int(term) for term in parameters['cytomine_id_terms'].split(',')]
		Xc = np.zeros((nims, len(term_list)))
		Yc = np.zeros(Xc.shape)
		i = 0
		for id_term in term_list:
			Xc[:, i] = xc[:, term_to_i[id_term]]
			Yc[:, i] = yc[:, term_to_i[id_term]]
			i += 1
	else:
		term_list = term_to_i.keys()
		Xc = xc
		Yc = yc
	(nims,nldms) = Xc.shape
	im_list = None
	if(parameters['cytomine_training_images'] != 'all'):
		im_list = [int(p) for p in parameters['cytomine_training_images'].split(',')]
	else:
		im_list = ims

	X = np.zeros((len(im_list),nldms))
	Y = np.zeros((len(im_list),nldms))
	im_to_i = {}
	for i in range(nims):
		im_to_i[ims[i]] = i

	for i in range(len(im_list)):
		X[i,:] = Xc[im_to_i[im_list[i]],:]
		Y[i,:] = Yc[im_to_i[im_list[i]],:]

	Xc = X
	Yc = Y


	h2 = generate_2_horizontal(parameters['model_W'], parameters['model_n'])
	v2 = generate_2_vertical(parameters['model_W'], parameters['model_n'])
	h3 = generate_3_horizontal(parameters['model_W'], parameters['model_n'])
	v3 = generate_3_vertical(parameters['model_W'], parameters['model_n'])
	sq = generate_square(parameters['model_W'], parameters['model_n'])

	joblib.dump((h2,v2,h3,v3,sq),'%s%s_lindner_feature_map.pkl'%(model_repo,parameters['model_name']))
	for id_term in term_list:
		(dataset,rep,img) = build_dataset_image_offset_mp(repository, Xc[:, term_to_i[id_term]], Yc[:, term_to_i[id_term]], im_list, parameters['model_D_MAX'], parameters['model_n_samples'], h2, v2, h3, v3, sq, parameters['model_njobs'])
		clf = VotingTreeRegressor(n_estimators=parameters['model_T'],n_jobs=parameters['model_njobs'])
		clf = clf.fit(dataset,rep)
		joblib.dump(clf, '%s%s_lindner_regressor_%d.pkl' % (model_repo, parameters['model_name'],id_term))

	xt = procrustes(Xc,Yc)
	(mu,P) = apply_pca(xt,parameters['model_n_reduc'])

	joblib.dump((mu,P),'%s%s_lindner_pca.pkl'%(model_repo,parameters['model_name']))

	F = open('%s%s_lindner_parameters.conf' % (parameters['model_save_to'], parameters['model_name']), 'wb')
	F.write('cytomine_id_terms %s\n' % parameters['cytomine_id_terms'])
	F.write('model_njobs %d\n' % parameters['model_njobs'])
	F.write('model_D_MAX %d\n' % parameters['model_D_MAX'])
	F.write('model_n_samples %d\n' % parameters['model_n_samples'])
	F.write('model_W %d\n' % parameters['model_W'])
	F.write('model_n %d\n' % parameters['model_n'])
	F.write('model_T %d\n' % parameters['model_T'])
	F.write('model_step %d\n' % parameters['model_step'])
	F.write('model_n_reduc %d\n' % parameters['model_n_reduc'])
	F.write('model_R_MAX %d\n' % parameters['model_R_MAX'])
	F.write('model_R_MIN %d\n' % parameters['model_R_MIN'])
	F.write('model_alpha %f\n' % parameters['model_alpha'])
	F.close()