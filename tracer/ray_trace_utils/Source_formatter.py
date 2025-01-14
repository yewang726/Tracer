import numpy as N

def get_source_vecs(filename):
	'''
	Originally used with Tonatuh data. Load directions and positions from a binary file.
	'''
	everything = N.fromfile(filename)

	nrays = len(everything)/6

	dx = everything[0::6]
	dy = -everything[2::6]
	dz = everything[1::6]

	directions = N.vstack([-dx, -dy, -dz])

	vx = everything[3::6]
	vy = -everything[5::6]
	vz = everything[4::6]

	vertices = N.vstack([vx, vy, vz])
	del(everything)
	return vertices, directions

def count_rays(list_source_files):
	pos = N.array([[],[],[]])
	dirs = N.array([[],[],[]])
	for s in xrange(len(list_source_files)):
		v,d = get_source_vecs(list_source_files[s])
		pos = N.concatenate((pos, v), axis=1)
		dirs = N.concatenate((dirs, d), axis=1)
	return N.shape(dirs)[1]

def format_for_optim(list_source_files, total_power, rays_per_source, target_dir):
	'''
	Originally used with Tonatuh data. Load a list of binary source files and transform them into a list of sources with equal energy and number of rays.
	'''
	pos = N.array([[],[],[]])
	dirs = N.array([[],[],[]])
	for s in xrange(len(list_source_files)):
		v,d = get_source_vecs(list_source_files[s])
		pos = N.concatenate((pos, v), axis=1)
		dirs = N.concatenate((dirs, d), axis=1)
	#print N.shape(dirs)
	source_number = 0
	number_of_sources = N.round(N.shape(pos)[1]/rays_per_source)

	for i in xrange(number_of_sources):

		#print source_number,'/', number_of_sources
		v_source = pos[:,i*rays_per_source:(i+1)*rays_per_source]
		d_source = dirs[:,i*rays_per_source:(i+1)*rays_per_source]
		ener = N.ones(rays_per_source)*total_power/rays_per_source
		v_out = N.ravel(v_source)
		d_out = N.ravel(d_source)
		X = N.hstack((v_out, d_out, ener))
		#N.save(file=target_dir+str(source_number), arr=X)
		X.tofile(file=target_dir+str(source_number))
		del(X)
		del(v_source)
		del(d_source)
		del(ener)
		source_number+=1

def format_Soltrace_data(source_files, total_power, rays_per_source, target_dir):
	'''
	Read a Soltrace raytrace data file and store the rays in binary source files with a given number of rays.
	'''
	pos = N.array([[],[],[]])
	dirs = N.array([[],[],[]])
	for s in source_files:
		data = N.loadtxt(s, delimiter=',', skiprows=1)
		receiver_hits = N.logical_and(data[:,6]==-1, data[:,7]==2)
		pos = N.concatenate((pos, data[receiver_hits, 0:3].T), axis=1)
		dirs = N.concatenate((dirs, data[receiver_hits, 3:6].T), axis=1)

	source_number = 0
	number_of_sources = int(N.round(pos.shape[1]/rays_per_source))
	for i in xrange(number_of_sources):

		#print source_number,'/', number_of_sources
		v_source = pos[:,i*rays_per_source:(i+1)*rays_per_source]
		d_source = dirs[:,i*rays_per_source:(i+1)*rays_per_source]
		ener = N.ones(rays_per_source)*total_power/rays_per_source
		v_out = N.ravel(v_source)
		d_out = N.ravel(d_source)
		X = N.hstack((v_out, d_out, ener))
		#N.save(file=target_dir+str(source_number), arr=X)
		X.tofile(file=target_dir+str(source_number))
		del(X)
		del(v_source)
		del(d_source)
		del(ener)
		source_number+=1

def format_Solstice_data(results_dir, n_rays_per_source=None, target_dir=None, option='S'):
	'''
	Read Solstice results data (solpaths and simulation result file) and store the rays in binary source files.
	option: 'S" to keep the starting location of the last ray of a path as the fixed point to export. 'E' to keep the end location of the last ray of the path top position the starting point of the ray at v1-d.
	'''
	if not n_rays_per_source is None:
		n_rays_per_source = float(n_rays_per_source)
		assert(target_dir is not None)
		from os import remove
		import glob

		files = glob.glob(target_dir+'/*')
		for f in files:
			remove(f)		

	resfile =  open(results_dir+'/simul', 'r')
	q_tot_target = float(resfile.readlines()[3].split(' ')[0])
	pathsfile = open(results_dir+'/solpaths', 'r')
	lines = pathsfile.readlines()
	pathsfile.close()
	
	n_verts = int(lines[5].split(' ')[1])
	n_rays =  int(lines[6+n_verts].split(' ')[1])
	vs = N.zeros((3, n_rays))
	ds = N.zeros((3, n_rays))

	idx_target = N.zeros(n_rays, dtype=bool)
	idx_occ = N.zeros(n_rays, dtype=bool)

	for i in xrange(n_rays):
		# Avoid the occluded and error paths:
		path_type = float(lines[6+n_verts+1+n_rays+3+i])
		# Get the starting vertex and the direction vector;
		last_seg = lines[6+n_verts+1+i].split(' ')[-2:]
		v0 = N.array(lines[6+int(last_seg[0])].split(' '), dtype=float)
		v1 = N.array(lines[6+int(last_seg[1])].split(' '), dtype=float)
		v = v1-v0
		ds[:,i] = v/N.sqrt(N.sum(v**2))
		if option =='S':
			vs[:,i] = v0
		if option == 'E':
			vs[:,i] = v1-ds[:,i]
		if path_type != 1.0:
			# Check if it is a target ray:
			if path_type == 0.5:
				idx_target[i] = True
		else:
			idx_occ[i] = True

	print 'occluded', N.sum(idx_occ), 'good', N.sum(idx_target), 'total', n_rays
	vs = vs[:,idx_target]
	ds = ds[:,idx_target]
	n_rays = N.sum(idx_target)#n_rays-N.sum(idx_occ)

	if n_rays_per_source is None:
		es = N.ones(n_rays)*q_tot_target/float(n_rays)
		binarize_source(vs, ds, es, target_dir+'/'+'0')

	else:
		n_sources = int(N.floor(n_rays/n_rays_per_source))
		rays_tot = int(n_sources*n_rays_per_source)
		vs = vs[:,:rays_tot]
		ds = ds[:,:rays_tot]
		es = N.ones(int(n_rays_per_source))*q_tot_target/n_rays_per_source

		for i in range(n_sources):
			id0, id1 = int(i*n_rays_per_source), int((i+1)*n_rays_per_source)
			binarize_source(vs[:,id0:id1], ds[:,id0:id1], es, target_dir+'/'+str(i))


def binarize_source(vs, ds, es, filename):
	'''
	Transform the bundle data into a list of source files with the relevant energy
	'''
	X = N.hstack((N.ravel(vs), N.ravel(ds), es))
	X.tofile(file=filename)
	del(X)

def load_source(fname):
	'''
	Load some already formatted and saved binary source data and converts it into position, direction and energy arrays.
	'''
	X = N.fromfile(fname)
	nrays = len(X)/7
	pos = N.vstack((X[0:nrays], X[nrays:2*nrays], X[2*nrays:3*nrays]))
	dirs = N.vstack((X[3*nrays:4*nrays], X[4*nrays:5*nrays], X[5*nrays:6*nrays]))
	ener = N.array(X[6*nrays:7*nrays])
	return pos, dirs, ener

def load_sources(list_fnames):
	'''
	Load some already formatted and saved binary sources data and converts it into position, direction and energy arrays. This function regroups arrays into single sources.
	'''
	pos = N.array([[],[],[]])
	dirs = N.array([[],[],[]])
	ener = N.array([])
	if not hasattr(list_fnames, 'len'): 
		list_fnames = list_fnames
	for fname in list_fnames:
		X = N.fromfile(fname)
		nrays = len(X)/7
		pos = N.concatenate((pos, N.vstack((X[0:nrays], X[nrays:2*nrays], X[2*nrays:3*nrays]))), axis=1)
		dirs = N.concatenate((dirs, N.vstack((X[3*nrays:4*nrays], X[4*nrays:5*nrays], X[5*nrays:6*nrays]))), axis=1)
		ener = N.concatenate((ener, N.array(X[6*nrays:7*nrays])))
	ener = ener/len(list_fnames)
	return pos, dirs, ener

def adjust_ener(fname, new_ener_tot):
	'''
	Adjust the energy on a previously saved source binary.
	'''
	pos, dirs, ener = load_source(fname)
	enernew = new_ener_tot/len(ener)*N.ones(len(ener))
	X = N.hstack((N.ravel(pos),N.ravel(dirs),enernew))
	print N.sum(ener), '->', N.sum(enernew)
	X.tofile(file=fname)

def push_back(fname, dist):
	'''
	Pushes rays back along their direction of propagation.
	'''
	pos,dirs,ener = load_source(fname)
	pos = pos-dist*dirs
	pos = N.ravel(pos)
	dirs = N.ravel(dirs)
	X = N.hstack((pos,dirs,ener))
	X.tofile(file=fname)

def push_to(fname, z):
	pos,dirs,ener = load_source(fname)
	t = (z-pos[2])/dirs[2]
	pos += t*dirs
	pos = N.ravel(pos)
	dirs = N.ravel(dirs)
	X = N.hstack((pos,dirs,ener))
	X.tofile(file=fname)


