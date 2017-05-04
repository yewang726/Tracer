import numpy as N
from numpy import random

from multiobjective_tools import *

import matplotlib.pyplot as plt

#from tracer.models.Two_N_parameters_cavity import *
from tracer.paraboloid import *
from tracer.models.SG4 import SG4
from tracer.tracer_engine import *
from tracer.tracer_engine_mp import *
from emissive_losses.emissive_losses import *

from tracer.CoIn_rendering.rendering import *

'''
Important features that need to be there: 
generate_scene()
get_objectives()
get_geometry()
self.termination_criteria
self.starting_geometry_boundaries
'''
from FONaR import *
from Source_formatter import *

class FONaR_ASME_optim():

	def __init__(self, termination_criteria, VF_precision, Tow_h, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, r_tube_ext, r_tube_int, k_t, rho_tube, T_in, T_out, T_amb, envelope_thickness, min_radius, max_radius, max_height, b_sphere_r, num_sections, angular_segments, tubes_per_element, cost_tubes, cost_insulation, sun_positions, num_rays, positions_weights):

		# Precision stopping criterion:	
		self.termination_criteria = termination_criteria
		self.VF_precision = VF_precision

		# Tower height:
		self.Tow_h = Tow_h

		# Radiative properties:
		self.absorptivity_absorber = absorptivity_absorber
		self.emissivity_absorber = emissivity_absorber
		self.absorptivity_envelope = absorptivity_envelope
		self.emissivity_envelope = emissivity_envelope

		# Tube parameters:
		self.r_tube_ext = r_tube_ext
		self.r_tube_int = r_tube_int
		self.k_t = k_t
		self.rho_tube = rho_tube

		# External convection coefficient:
		self.u_conv_ext = 30.

		# Temperatures:
		self.T_in = T_in
		self.T_out = T_out
		self.T_amb = T_amb

		# Envelope fixed thickness:
		self.envelope_thickness = envelope_thickness

		# Parameter space limits:
		self.num_sections = num_sections
		self.angular_segments = angular_segments
		self.tubes_per_element = tubes_per_element

		self.max_radius = max_radius
		self.max_height = max_height
		self.b_sphere_r = b_sphere_r

		min_rs, max_rs = N.ones(self.num_sections+1)*min_radius, N.ones(self.num_sections+1)*max_radius
		min_zs, max_zs = N.zeros(self.num_sections+1)+self.envelope_thickness, N.ones(self.num_sections+1)*max_height-self.envelope_thickness
		# FIXME!!! to prevent shit happening!!!
		min_zs[0] = 0.
		max_zs[0] = 0.
		self.starting_geometry_boundaries = [[N.array([min_rs, min_zs]), N.array([max_rs, max_zs])]]

		# Materials costs:
		self.cost_tubes = cost_tubes #/m
		self.cost_insulation = cost_insulation #/m2

		# Sources directories and weighting factors, one per position:
		self.sun_positions = sun_positions
		self.num_rays = num_rays
		self.positions_weights = N.array(positions_weights)
		self.positions_weights = self.positions_weights/N.sum(self.positions_weights)


	def generate_geom(self, geom_mins=None, geom_maxs=None):

		# radii, heights, envelope_thickness, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, location=None, rotation=None

		if (geom_mins == None) and (geom_maxs == None):
			min_rs, max_rs = self.starting_geometry_boundaries[0][0][0], self.starting_geometry_boundaries[0][1][0]
			min_zs, max_zs = self.starting_geometry_boundaries[0][0][1], self.starting_geometry_boundaries[0][1][1]
		else:
			min_rs, max_rs = geom_mins[0], geom_maxs[0]
			min_zs, max_zs = geom_mins[1], geom_maxs[1]

		rs = N.zeros(self.num_sections+1)
		zs = N.zeros(self.num_sections+1)

		# Create position vectors:
		p = N.zeros((2,self.num_sections)) # N.vstack((rs,zs))
		# Create direction vectors:
		d = N.zeros((2,self.num_sections)) # N.vstack((rs[1:]-rs[:-1], zs[1:]-zs[:-1])/norms)
		# Norms:
		norms = N.zeros(self.num_sections)

		searching = True
		while searching:
			for r in xrange(1, self.num_sections-2):
				rs[r+1] = N.random.uniform(min_rs[r+1], max_rs[r+1])
			for r in xrange(self.num_sections-1):
				zs[r+1] = N.random.uniform(min_zs[r+1], max_zs[r+1])
			#zs = N.sort(zs) # Upward geometry sections only.

			# check the obtained geometry for constraints relating to the bounding box.
			p[:,r] = rs[r],zs[r]
			norms[r] = N.sqrt((rs[r+1]-rs[r])**2.+(zs[r+1]-zs[r])**2.)
			d[:,r] = (rs[r+1]-rs[r])/norms[r], (zs[r+1]-zs[r])/norms[r]

			center_sphere_z = zs[-1]/2.
			r_sphere = N.sqrt(rs**2.+(zs-center_sphere_z)**2.)
			if (r_sphere<self.b_sphere_r).all(): #all the shape is in the spherical bounding box:
				rs_bad = False
			# Test intersections: td+v=0.
			# Intersect every line with all others in the scene. If intersections are outside of the parameters for ech side, no intersection and the geometry is good to go. Otherwise, re-iterate.
			for v1 in xrange(N.shape(p)[1]):
				v2 = (N.arange(N.shape(p)[1])-v1) != 0

				t1 = (d[0,v2]*(p[1,v2]-p[1,v1])-d[1,v2]*(p[0,v2]-p[0,v1]))/(d[1,v1]*d[0,v2]-d[0,v1]*d[1,v2])
				t2 = (p[0,v1]-p[0,v2]+t1*d[0,v1])/d[0,v2]

				# If any intersection, the parameters are in the range [0., norm] for both segments:
				if N.logical_and(N.logical_and(t1>0.,t1<norms[v1]), N.logical_and(t2>0., t2<norms[v2])).any():
					self_cutting = True
					break
				else:
					self_cutting = False
			searching = rs_bad and self_cutting
		print 'Geometry found: ', rs, zs
		radii = rs
		heights = zs[1:]-zs[:-1]
		location=N.array([0.,0.,self.aim_h])#-self.L_abs/2.])

		return radii, heights, location, rs, zs

	def generate_scene(self, geom_mins=None, geom_maxs=None):

		radii, heights, location, rs, zs, norms = self.generate_geom(geom_mins, geom_maxs)

		receiver_params = [radii, heights, self.envelope_thickness, self.absorptivity_absorber, self.emissivity_absorber, self.absorptivity_envelope, self.emissivity_envelope, location]
		# Discretise and generate the VF_matrix.
		rec = FONaR(*receiver_params)

		# Discretisation parameters:
		angular_bins = self.angular_segments
		vb = N.rint(norms/self.elems_len)
		vertical_bins = vb.astype(int)
		envelope_vertical_bins = 1
		self.discretisation_params = [angular_bins, vertical_bins, envelope_vertical_bins]

		rec.discretise(*self.discretisation_params)
		bins_breakdown = [rec.angular_bins, rec.vertical_bins, rec.envelope_vertical_bins]
		# Initialise MCRT bins:
		rec.bins = N.zeros((len(self.sun_positions),len(rec.areas[3:])))
		# Initialise spillage value:
		rec.spillage = N.zeros(len(self.sun_positions))
		print 'Geometry discretised into %s elements.'%(str(N.shape(rec.binning_scheme[3:])[0]))
		rec.VF = None

		return [receiver_params, rec.binning_scheme, rec.VF, rec.areas, rec.absorptivities, rec.emissivities, bins_breakdown, rec.bins, rec.spillage]

	def get_objectives(self, params, source_idx, procs=8):

		#E_eff_tot, T_hom_tot, cost_tot = 0., 0., 0.
		E_eff_tot, T_max_tot, mass_tot = 0., 0., 0.
		
		for sun_pos in xrange(len(self.sun_positions)):
			print sun_pos+1,'/',len(self.sun_positions),' sun positions raytraced.'
			receiver = self.build_receiver(params, sun_pos)
			# Build the source array. Import a sphere of rays from the current position and split it for multi-processing
			sources = []
			p,d,e = load_source(self.sun_positions[sun_pos]+str(source_idx))
			nr_p_s = len(e)/procs
			for s in xrange(procs):
				sources.append(RayBundle(p[:,s*nr_p_s:(s+1)*nr_p_s], d[:,s*nr_p_s:(s+1)*nr_p_s], e[s*nr_p_s:(s+1)*nr_p_s]))

			# Ray-trace:
			engine =TracerEngineMP(receiver)
			engine.multi_ray_sim(sources, procs=procs, minener=1e-15, reps=1000 , tree=True)
			receiver = engine._asm
			receiver.spillage = N.sum(engine.tree._bunds[0].get_energy())-N.sum(engine.tree._bunds[0].get_energy(engine.tree._bunds[1].get_parents()))


			'''
			from CoIn_rendering.rendering import Renderer
			view = Renderer(engine)
			view.show_rays()
			#'''

			# Energy balance:
			bins = receiver.bin_hits()
			receiver.bins = (receiver.bins*source_idx+bins)/(source_idx+1.)
			params[7][sun_pos,:] = receiver.bins
			params[8][sun_pos] = receiver.spillage
			receiver = self.get_balance(receiver)

			if receiver.bad_geom == True:
 				E_eff_tot,T_hom_tot,cost_tot = N.nan, N.nan, N.nan
				break
			else:
				# Objectives:
				# Energy efficiency:
				receiver.q_source = N.sum(e)-receiver.spillage
				#E_eff = N.sum(receiver.q_net-receiver.q_conv_ext)/receiver.q_source
				E_eff = N.sum(receiver.q_net-receiver.q_conv_ext)/N.sum(e)
				# Temperature homogeneity:
				env_len = receiver.envelope_vertical_bins*receiver.angular_bins
				T_abs = receiver.T[receiver.flow_path]
				ls = N.add.accumulate(N.hstack((0.,receiver.tube_lengths)))
				#T_hom = 1./(1.+N.sum(N.abs((T_abs[1:]**2.-T_abs[:-1]**2.))/(ls[2:]-ls[:-2]))/2./(self.T_out-self.TN.array([E_eff_tot, T_hom_tot, cost_tot])_in)**2.)
				T_max = N.amax(T_abs)/self.T_out
				# Cost:
				Rbot = receiver.bottom_radius[0]
				Rtop = receiver.top_radius[-1]
				R = Rbot+(N.arange(N.sum(receiver.vertical_bins))+0.5)*(Rtop-Rbot)/N.sum(receiver.vertical_bins)
				R_tube = R-receiver.r_tube_e*N.sqrt(1-((Rtop-Rbot)/N.sum(receiver.height))**2.)

				if env_len != 0.:
					A_insulation = N.sum(receiver.areas[3:(env_len+3)]+receiver.areas[-env_len:])
				else:
					A_insulation = 0.
				A_ins_max = self.max_radius*self.envelope_thickness*2.*N.pi
				l_t_max = self.max_radius*self.max_height*N.pi/self.r_tube_ext
				m_t_max =  l_t_max*N.pi*(self.r_tube_ext**2.-self.r_tube_int**2.)*self.rho_tube
				mass_tubes = N.sum(receiver.tube_lengths)*N.pi*(self.r_tube_ext**2.-self.r_tube_int**2.)*self.rho_tube
				#cost = 1./(1.+(self.cost_tubes*mass_tubes+self.cost_insulation*A_insulation)/(self.cost_tubes*m_t_max+self.cost_insulation*A_ins_max))

				E_eff_tot = E_eff_tot+E_eff*self.positions_weights[sun_pos]
				#T_hom_tot = T_hom_tot+T_hom*self.positions_weights[sun_pos]
				#cost_tot = cost
				T_max_tot = 1./(1.+N.amax((T_max_tot,T_max)))
				mass_tot = 1./(1.+mass_tubes)

		self.params = params
		return N.array([E_eff_tot, T_max_tot, mass_tot])

	def get_geometry(self, params):
		receiver_params = params[0]
		radii = receiver_params[0]
		heights = receiver_params[1]
		envelope_thickness = receiver_params[2]
		rs = radii
		zs = envelope_thickness+N.hstack(([0],N.add.accumulate(heights)))+receiver_params[-1][2]-self.Tow_h
		return N.vstack((rs, zs))

	def build_receiver(self, params, sun_pos):
		# Build the scene from the params:
		receiver = FONaR(*params[0])
		receiver.binning_scheme = params[1]
		receiver.VF = params[2]
		receiver.areas = params[3]
		receiver.absorptivities = params[4]
		receiver.emissivities = params[5]
		receiver.angular_bins = params[6][0]
		receiver.vertical_bins = params[6][1]
		receiver.envelope_vertical_bins = params[6][2]
		receiver.bins = params[7][sun_pos,:]
		receiver.spillage = params[8][sun_pos]

		return receiver

	def get_balance(self, receiver):
		receiver.ASME_balance(bins=receiver.bins, T_in=self.T_in, T_out=self.T_out, r_tube_e=self.r_tube_ext, r_tube_in=self.r_tube_int, tubes_per_bank=self.tubes_per_element, k_tube=self.k_tube, u_conv_ext=self.u_conv_ext, T_amb = 293.15)

		return receiver

	def get_sources_params(self, sun_pos, source_idxs):
		p = N.array([[],[],[]])
		d = N.array([[],[],[]])
		e = N.array([])
		for i in xrange(len(source_idxs)):
			pi,di,ei = load_source(self.sun_positions[sun_pos]+str(source_idxs[i]))
			p = N.concatenate((p, pi), axis=1)
			d = N.concatenate((d, di), axis=1)
			e = N.concatenate((e, ei))
		return p,d,e


class FONaR_SP_optim():

	def __init__(self, termination_criteria, VF_precision, aim_h, R_aim, L_abs, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, envelope_thickness, min_radius, max_radius, b_sphere_r, num_sections, angular_segments, v_sections_per_element, sun_positions, num_rays, positions_weights):

		# Precision stopping criterion:	
		self.termination_criteria = termination_criteria
		self.VF_precision = VF_precision

		# Fixed params:
		self.aim_h= aim_h
		self.R_aim = R_aim
		self.L_abs = L_abs

		# Radiative properties:
		self.absorptivity_absorber = absorptivity_absorber
		self.emissivity_absorber = emissivity_absorber
		self.absorptivity_envelope = absorptivity_envelope
		self.emissivity_envelope = emissivity_envelope

		# External convection coefficient:
		self.u_conv_ext = 30.

		# Envelope fixed thickness:
		self.envelope_thickness = envelope_thickness

		# Parameter space limits:
		self.num_sections = num_sections
		self.angular_segments = angular_segments
		self.v_sections_per_element = v_sections_per_element

		self.max_radius = max_radius
		self.b_sphere_r = b_sphere_r

		min_rs, max_rs = N.ones(self.num_sections+1)*min_radius, N.ones(self.num_sections+1)*max_radius
		min_zs, max_zs = N.zeros(self.num_sections+1)+self.envelope_thickness, N.ones(self.num_sections+1)*L_abs-self.envelope_thickness
		self.starting_geometry_boundaries = [[N.array([min_rs, min_zs]), N.array([max_rs, max_zs])]]

		# Sources directories and weighting factors, one per position:
		self.sun_positions = sun_positions
		self.num_rays = num_rays
		self.positions_weights = N.array(positions_weights)
		self.positions_weights = self.positions_weights/N.sum(self.positions_weights)

	def generate_geom(self, geom_mins=None, geom_maxs=None):

		# radii, heights, envelope_thickness, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, location=None, rotation=None

		if (geom_mins == None) and (geom_maxs == None):
			min_rs, max_rs = self.starting_geometry_boundaries[0][0][0], self.starting_geometry_boundaries[0][1][0]
			min_zs, max_zs = self.starting_geometry_boundaries[0][0][1], self.starting_geometry_boundaries[0][1][1]
		else:
			min_rs, max_rs = geom_mins[0], geom_maxs[0]
			min_zs, max_zs = geom_mins[1], geom_maxs[1]

		rs = N.zeros(self.num_sections+1)
		zs = N.zeros(self.num_sections+1)

		# SP study constraints
		rs[:2] = self.R_aim
		rs[-2:] = self.R_aim
		zs[-1] = self.L_abs

		# Create position vectors:
		p = N.zeros((2,self.num_sections)) # N.vstack((rs,zs))
		# Create direction vectors:
		d = N.zeros((2,self.num_sections)) # N.vstack((rs[1:]-rs[:-1], zs[1:]-zs[:-1])/norms)
		# Norms:
		norms = N.zeros(self.num_sections)

		searching = True
		while searching:
			for r in xrange(1, self.num_sections-2):
				rs[r+1] = N.random.uniform(min_rs[r+1], max_rs[r+1])
			for r in xrange(self.num_sections-1):
				zs[r+1] = N.random.uniform(min_zs[r+1], max_zs[r+1])
			zs = N.sort(zs) # Upward geometry sections only.

			# check the obtained geometry for constraints relating to the bounding box.
			p[:,r] = rs[r],zs[r]
			norms[r] = N.sqrt((rs[r+1]-rs[r])**2.+(zs[r+1]-zs[r])**2.)
			d[:,r] = (rs[r+1]-rs[r])/norms[r], (zs[r+1]-zs[r])/norms[r]

			center_sphere_z = zs[-1]/2.
			r_sphere = N.sqrt(rs**2.+(zs-center_sphere_z)**2.)
			if (r_sphere<self.b_sphere_r).all(): #all the shape is in the spherical bounding box:
				rs_bad = False
			# Test intersections: td+v=0.
			# Intersect every line with all others in the scene. If intersections are outside of the parameters for ech side, no intersection and the geometry is good to go. Otherwise, re-iterate.
			for v1 in xrange(N.shape(p)[1]):
				v2 = (N.arange(N.shape(p)[1])-v1) != 0

				t1 = (d[0,v2]*(p[1,v2]-p[1,v1])-d[1,v2]*(p[0,v2]-p[0,v1]))/(d[1,v1]*d[0,v2]-d[0,v1]*d[1,v2])
				t2 = (p[0,v1]-p[0,v2]+t1*d[0,v1])/d[0,v2]

				# If any intersection, the parameters are in the range [0., norm] for both segments:
				if N.logical_and(N.logical_and(t1>0.,t1<norms[v1]), N.logical_and(t2>0., t2<norms[v2])).any():
					self_cutting = True
					break
				else:
					self_cutting = False
			searching = rs_bad and self_cutting
		print 'Geometry found: ', rs, zs
		radii = rs
		heights = zs[1:]-zs[:-1]
		location=N.array([0.,0.,self.aim_h])#-self.L_abs/2.])

		return radii, heights, location, rs, zs

	def generate_scene(self, geom_mins=None, geom_maxs=None):

		radii, heights, location, rs, zs = self.generate_geom(geom_mins, geom_maxs)

		receiver_params = [radii, heights, self.envelope_thickness, self.absorptivity_absorber, self.emissivity_absorber, self.absorptivity_envelope, self.emissivity_envelope, location]

		# Discretise and generate the VF_matrix.
		rec = FONaR(*receiver_params)

		# Discretisation parameters:
		angular_bins = self.angular_segments
		vertical_bins = N.ones(self.num_sections, dtype=int)*self.v_sections_per_element
		envelope_vertical_bins = 1
		self.discretisation_params = [angular_bins, vertical_bins, envelope_vertical_bins]

		rec.discretise(*self.discretisation_params)
		bins_breakdown = [rec.angular_bins, rec.vertical_bins, rec.envelope_vertical_bins]
		# Initialise MCRT bins:
		rec.bins = N.zeros((len(self.sun_positions),len(rec.areas[3:])))
		# Initialise spillage value:
		rec.spillage = N.zeros(len(self.sun_positions))
		print 'Geometry discretised.'
		if self.VF_precision == 'convex':
			rec.VF = N.zeros((N.shape(rec.binning_scheme)[0],N.shape(rec.binning_scheme)[0]))
			rec.VF[1,3:] = rec.areas[3:]/N.sum(rec.areas[:3])
			rec.VF[1,1] = 1.-N.sum(rec.VF[1,3:])
			rec.VF[3:,1] = 1.
		else:
			print 'VF calculation...'
			rec.VF_sim(num_rays=10000, precision=self.VF_precision)
		print 'VF matrix complete'
		print ' '

		return [receiver_params, rec.binning_scheme, rec.VF, rec.areas, rec.absorptivities, rec.emissivities, bins_breakdown, rec.bins, rec.spillage]

	def get_objectives(self, params, source_idx, procs=8):

		MC_eff_tot, T_hom_tot, area_tot = 0., 0., 0.
		for sun_pos in xrange(len(self.sun_positions)):
			print sun_pos+1,'/',len(self.sun_positions),' sun positions raytraced.'
			receiver = self.build_receiver(params, sun_pos)

			# Build the source array. Import a sphere of rays from the current position and split it for multi-processing
			sources = []
			p,d,e = load_source(self.sun_positions[sun_pos]+str(source_idx))
			nr_p_s = len(e)/procs
			for s in xrange(procs):
				sources.append(RayBundle(p[:,s*nr_p_s:(s+1)*nr_p_s], d[:,s*nr_p_s:(s+1)*nr_p_s], e[s*nr_p_s:(s+1)*nr_p_s]))

			# Ray-trace:
			engine =TracerEngineMP(receiver)
			engine.multi_ray_sim(sources, procs=procs, minener=1e-15, reps=1000 , tree=True)
			receiver = engine._asm
			receiver.spillage = N.sum(engine.tree._bunds[0].get_energy())-N.sum(engine.tree._bunds[0].get_energy(engine.tree._bunds[1].get_parents()))

			# Single position objective determination:
			bins = receiver.bin_hits()
			receiver.bins = (receiver.bins*source_idx+bins)/(source_idx+1.)
			params[7][sun_pos,:] = receiver.bins
			params[8][sun_pos] = receiver.spillage

			receiver.MC_eff(receiver.bins)
			
			# Objectives:
			# Energy efficiency:
			receiver.q_source = N.sum(e)-receiver.spillage
			MC_eff_tot = N.sum(receiver.MC_effs*receiver.q_in*receiver.areas[3:])/N.sum(e)
			# Temperature homogeneity:
			T = receiver.T

			T_hom_tot = 1./(1.+N.sum(receiver.areas[3:]*T)/N.sum(receiver.areas[3:]))
			# area:
			area_tot = 1./(1.+N.sum(receiver.areas[3:]))

		self.params = params
		return N.array([MC_eff_tot, T_hom_tot, area_tot])

	def get_MC_eff(self, receiver, Tlim=None):
		receiver.MC_eff(Tlim=Tlim)
		return receiver

	def get_geometry(self, params):
		receiver_params = params[0]
		radii = receiver_params[0]
		heights = receiver_params[1]
		envelope_thickness = receiver_params[2]
		rs = radii
		zs = envelope_thickness+N.hstack(([0],N.add.accumulate(heights)))+receiver_params[-1][2]-self.aim_h
		return N.vstack((rs, zs))

	def build_receiver(self, params, sun_pos):
		# Build the scene from the params:
		receiver = FONaR(*params[0])
		receiver.binning_scheme = params[1]
		receiver.VF = params[2]
		receiver.areas = params[3]
		receiver.absorptivities = params[4]
		receiver.emissivities = params[5]
		receiver.angular_bins = params[6][0]
		receiver.vertical_bins = params[6][1]
		receiver.envelope_vertical_bins = params[6][2]
		receiver.bins = params[7][sun_pos,:]
		receiver.spillage = params[8][sun_pos]
		return receiver

	def get_sources_params(self, sun_pos, source_idxs):
		p = N.array([[],[],[]])
		d = N.array([[],[],[]])
		e = N.array([])
		for i in xrange(len(source_idxs)):
			pi,di,ei = load_source(self.sun_positions[sun_pos]+str(source_idxs[i]))
			p = N.concatenate((p, pi), axis=1)
			d = N.concatenate((d, di), axis=1)
			e = N.concatenate((e, ei))
		return p,d,e

class FONaR_These_optim(FONaR_ASME_optim):
	'''
	An optimisation with three objectives:
	- Second law efficiency based on the hydrodynamic model and the flow path
	- Lowest maximum thermal stress with some cutoff threshold for the limit stress
	- Minimal mass of pipes
	'''
	def __init__(self, termination_criteria, objectives_thresholds, VF_precision, Tow_h, absorptivity_absorber, emissivity_absorber, r_tube_ext, r_tube_int, coating_thickness, k_coating, material, HC, T_in, T_out, T_amb, min_radius, max_radius, max_height, b_sphere_r, num_sections, flow_paths, angular_segments, tubes_per_element, sun_positions, num_rays, positions_weights):
		envelope_thickness = 0
		absorptivity_envelope = 1.
		emissivity_envelope = 1.
		cost_tubes = 1.
		cost_insulation = 1.
		self.HC = HC
		self.material = material
		self.rho_tube = material.rho()
		self.objectives_thresholds = objectives_thresholds
		self.flow_paths = flow_paths

		self.coating_thickness = coating_thickness
		self.k_coating = k_coating

		self.elems_len = 2.*(r_tube_ext+coating_thickness)*tubes_per_element
		k_t = 0.
		rho_tube = 0.

		FONaR_ASME_optim.__init__(self, termination_criteria, VF_precision, Tow_h, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, r_tube_ext, r_tube_int, k_t, rho_tube, T_in, T_out, T_amb, envelope_thickness, min_radius, max_radius, max_height, b_sphere_r, num_sections, angular_segments, tubes_per_element, cost_tubes, cost_insulation, sun_positions, num_rays, positions_weights)


	def generate_geom(self, geom_mins=None, geom_maxs=None):

		# radii, heights, envelope_thickness, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, location=None, rotation=None

		if (geom_mins == None) and (geom_maxs == None):
			min_rs, max_rs = self.starting_geometry_boundaries[0][0][0], self.starting_geometry_boundaries[0][1][0]
			min_zs, max_zs = self.starting_geometry_boundaries[0][0][1], self.starting_geometry_boundaries[0][1][1]
		else:
			min_rs, max_rs = geom_mins[0], geom_maxs[0]
			min_zs, max_zs = geom_mins[1], geom_maxs[1]

		rs = N.zeros(self.num_sections+3)
		zs = N.zeros(self.num_sections+3)

		searching = True
		while searching:
			for r in xrange(self.num_sections+1):
				#if (r+1)<self.num_sections:
				rs[r+1] = N.random.uniform(min_rs[r], max_rs[r])
				if r>0:
					zs[r+1] = N.random.uniform(min_zs[r], max_zs[r])
					#zs = N.sort(zs) # Upward geometry sections only.
					if self.elems_len != None:
						norm = N.sqrt((rs[r+1]-rs[r])**2.+(zs[r+1]-zs[r])**2.)
						dloc = N.vstack(((rs[r+1]-rs[r])/norm, (zs[r+1]-zs[r])/norm))
						norm_r = N.ceil(norm/self.elems_len)*self.elems_len

						rs[r+1] = rs[r]+(rs[r+1]-rs[r])*norm_r/norm
						zs[r+1] = zs[r]+(zs[r+1]-zs[r])*norm_r/norm

			# Build the top and bottom caps artificially to avoid intersections.
			rs[-1] = 0.
			zs[-1] = zs[-2]

			center_sphere_z = zs[-1]/2.
			r_sphere = N.sqrt(rs**2.+(zs-center_sphere_z)**2.)
			rs_bad = True
			zs_bad = True
			sphere_ok = (r_sphere<self.b_sphere_r).all()
			rs_ok = (rs[1:-1]>min_rs).all() and (rs[1:-1]<max_rs).all()
			if sphere_ok and rs_ok: #all the shape is in the spherical bounding box:
				rs_bad = False
			top_at_top = (zs[-2]>=zs[:-2]).all()
			top_ok = zs[-1]<max_zs[-1]
			bot_ok = N.amin(zs)>=0.
			if top_at_top and top_ok and bot_ok:
				zs_bad = False # The bottom of the receiver is at the bottom

			# Test intersections: td+v=0.
			# check the obtained geometry for constraints relating to the bounding box.
			p = N.vstack((rs,zs))
			norms = N.sqrt((rs[1:]-rs[:-1])**2.+(zs[1:]-zs[:-1])**2.)
			d = N.vstack(((rs[1:]-rs[:-1])/norms, (zs[1:]-zs[:-1])/norms))

			# Intersect every line with all others in the scene. If intersections are outside of the parameters for ech side, no intersection and the geometry is good to go. Otherwise, re-iterate.
			for v1 in xrange(N.shape(p)[1]-1):
				v2 = N.arange(N.shape(p)[1]-1)!=v1

				t1 = (d[0,v2]*(p[1,v2]-p[1,v1])-d[1,v2]*(p[0,v2]-p[0,v1]))/(d[1,v1]*d[0,v2]-d[0,v1]*d[1,v2])
				t2 = (p[0,v1]-p[0,v2]+t1*d[0,v1])/d[0,v2]

				# If any intersection, the parameters are in the range [0., norm] for both segments:
				self_cutting = False		
				if N.logical_and(N.logical_and(t1>0.,t1<(norms[v1])), N.logical_and(t2>0., t2<(norms[v2]))).any():
					self_cutting = True
					#print 'Cut!'
					break
			
			searching = rs_bad or self_cutting or zs_bad
			#print searching
		# Get the profile only and forget about the caps.
		rs = rs[1:-1]
		zs = zs[1:-1]
		norms = norms[1:-1]
		print 'Geometry found: ', rs, zs, norms
		radii = rs
		heights = zs[1:]-zs[:-1]
		location=N.array([0.,0.,self.Tow_h])

		return radii, heights, location, rs, zs, norms

	def get_balance(self, receiver):

		receiver.thesis_balance(receiver.bins, T_in=self.T_in, T_out=self.T_out, r_tube_e=self.r_tube_ext, r_tube_in=self.r_tube_int, tube_material=self.material, HC=self.HC, u_conv_ext=self.u_conv_ext, T_amb=self.T_amb, k_coating=self.k_coating, coating_thickness=self.coating_thickness, gap_between_tubes=None, fp_option='mhtb%s'%(str(self.flow_paths)))

		return receiver

	def get_objectives(self, params, source_idx, procs=8):

		E_eff_tot, S_max_tot, mass_tot = 0., 0., 0.
		F_max_tots = N.zeros(len(self.sun_positions))
		
		for sun_pos in xrange(len(self.sun_positions)):
			print sun_pos+1,'/',len(self.sun_positions),' sun positions raytraced.'
			receiver = self.build_receiver(params, sun_pos)
			# Build the source array. Import a sphere of rays from the current position and split it for multi-processing
			sources = []
			p,d,e = load_source(self.sun_positions[sun_pos]+str(source_idx))
			nr_p_s = len(e)/procs
			for s in xrange(procs):
				sources.append(RayBundle(p[:,s*nr_p_s:(s+1)*nr_p_s], d[:,s*nr_p_s:(s+1)*nr_p_s], e[s*nr_p_s:(s+1)*nr_p_s]))
			source_ener = N.sum(e)

			# Ray-trace:
			engine =TracerEngineMP(receiver)
			engine.multi_ray_sim(sources, procs=procs, minener=1e-15, reps=1000, tree=True)
			receiver = engine._asm

			botenv = N.sum(engine._asm.get_objects()[0].get_surfaces()[0].get_optics_manager().get_all_hits()[0])
			topenv = N.sum(engine._asm.get_objects()[2].get_surfaces()[0].get_optics_manager().get_all_hits()[0])
			qcon = N.sum(engine.tree._bunds[0].get_energy())
			qhitting =  N.sum(engine.tree._bunds[0].get_energy(engine.tree._bunds[1].get_parents()))

			spillage = qcon-qhitting+botenv+topenv

			#print 'Qcon:', N.sum(engine.tree._bunds[0].get_energy())

			receiver.spillage = (spillage+receiver.spillage*source_idx)/(source_idx+1.)
			params[8][sun_pos] = receiver.spillage

			'''
			from CoIn_rendering.rendering import Renderer
			view = Renderer(engine)
			view.show_rays()
			#'''

			# Energy balance:
			if receiver.spillage>(0.5*source_ener): # spillage filter to avoid geometries that are spilling a lot to get through and mess-up the radiosity matrix inversion.
				receiver.bad_geom = True
			else:
				bins = receiver.bin_hits()
				receiver.bins = (receiver.bins*source_idx+bins)/(source_idx+1.)
				params[7][sun_pos,:] = receiver.bins
				if receiver.VF == None:			
					if self.VF_precision == 'convex':
						receiver.VF = N.zeros((N.shape(rec.binning_scheme)[0],N.shape(rec.binning_scheme)[0]))
						receiver.VF[1,3:] = rec.areas[3:]/N.sum(rec.areas[:3])
						receiver.VF[1,1] = 1.-N.sum(rec.VF[1,3:])
						receiver.VF[3:,1] = 1.
					else:
						print 'VF calculation...'
						receiver.VF_sim(num_rays=5000, precision=self.VF_precision)
					print 'VF matrix complete'
					print ' '
					params[2] = receiver.VF

				receiver = self.get_balance(receiver)

			if receiver.bad_geom == True:
 				E_eff_tot, F_max_tot, cost_tot = N.nan, N.nan, N.nan
				break
			else:
				# Objectives:
				# Work efficiency:
				pdropwork = []
				for f in xrange(len(receiver.fp)):
					rhos = receiver.HC.rho(receiver.T_HC[f])
					pdropwork.append(N.sum(receiver.Dp[receiver.fp[f]]/(rhos[1:]+rhos[:-1]))*2.*receiver.m[f])
				E_eff = (N.sum(receiver.q_net)*(1.-self.T_amb/self.T_out)-N.sum(pdropwork))/N.sum(e)
				E_eff_tot = E_eff_tot+E_eff*self.positions_weights[sun_pos]
				'''
				# Maximum stress:
				q = receiver.q_net/receiver.areas[3:]
				T = receiver.T_w_ext
				a = receiver.material.alpha(T)
				E = receiver.material.E(T)
				k = receiver.material.k(T)
				nu = receiver.material.nu(T)
				#su_conv_int = receiver.u_conv_int

				#stress = N.abs(flux_net*alpha*E/N.pi*(t_w/(2.*k_t)*((N.pi*(2.-nu_t)-(1.-nu_t))/(1.-nu_t))+(N.pi-1.)/u_conv_int))

				stress = N.abs(q*self.r_tube_ext*a*E/(2.*k*(1.-nu))*(1.-2.*self.r_tube_int**2./(self.r_tube_ext**2.-self.r_tube_int**2.)*N.log(self.r_tube_ext/self.r_tube_int)))
				UTS = receiver.material.UTS(T)
				stress_max = N.amax(stress/UTS)
				S_max_tots[sun_pos] = 1./(1.+stress_max)
				'''
				# Maximum flux:
				fmax = N.amax(receiver.bins/receiver.areas[3:])/1e6
				F_max_tots[sun_pos] = 1./(1.+fmax)

				# Mass of tubes:
				mass_tubes = N.sum(N.hstack(receiver.elem_lengths)*receiver.n_tubes*N.pi/4.*(receiver.D_tubes_e**2.-receiver.D_tubes_i**2.)*receiver.material.rho())
				mass_tot = 1./(1.+mass_tubes/10000.)

				if N.isnan(N.array([E_eff_tot, S_max_tot, mass_tot])).any(): # the objectives spit out a nan that could not be caught before (mostly due to material limitations).
					receiver.bad_geom = True
					break
		#S_max_tot = N.amin(S_max_tots)
		F_max_tot = N.amin(F_max_tots)
		self.params = params

		return N.array([E_eff_tot, F_max_tot, mass_tot])



class USASEC_optim():
	# Class to initialise the scenes with the desired receiver models and compute the objectives from the ray-traces 

	def __init__(self):

		# Receiver parameters _________________________________________________________________
		# Geometry:
		self.max_r = 0.65
		self.max_d = 1.5
		self.N_elems = 3
		self.min_r = 0.1
		self.min_d = 0.

		self.wall_insu = 0.1
		self.cone_cut_radius = None
		self.passive_cone = False
		self.pancake = 1
		self.elements_per_section = 10
		# Optical properties:
		self.specular_receiver = False
		absReceiver = 0.95
		emsReceiver = 0.85
		# Effective radiative properties values (taking into account tube covered surfaces):
		self.absReceiver = absReceiver/(2./N.pi*(1.-absReceiver)+absReceiver)
		if emsReceiver == None:	
			self.emsReceiver = absReceiver # Receiver emissivity: gray body assumption
		else:
			self.emsReceiver = emsReceiver/(2./N.pi*(1.-emsReceiver)+emsReceiver)

		min_rs = self.min_r*N.ones(self.N_elems+1)
		if self.cone_cut_radius!=None:
			min_rs[0] = self.cone_cut_radius
		min_rs[-1] = 0.
		max_rs=self.max_r*N.ones(self.N_elems+1)
		max_rs[0] = max_rs[0]+self.wall_insu
		max_rs[-1] = 0.
		min_xs = self.min_d-self.max_d
		max_xs = self.max_d

		self.starting_geometry_boundaries = [N.array([min_rs, min_xs]), N.array([max_rs,max_xs])]

		# envelope constraints (Check that on the Two_N_parammeters_cavity):
		self.envelope_radius = None
		self.envelope_depth = None
		# HC loop properties:
		self.Tamb = 300.
		self.Trec_in = 273.15+60.
		self.p_in = 5e6
		self.Trec_out = 273.15+500.
		self.tube_diameters_in = 0.015802
		self.tube_diameters_out = 0.021340
		self.tube_conductivity = 20.
		self.passive = None
		if self.cone_cut_radius!=None or self.passive_cone != False:
			self.N_elems += 1
			if self.passive_cone != False:
				self.passive = N.arange(-self.elements_per_section,0)

		# Optics parameters ____________________________________________________________________
		self.dishDiameter = 24.88
		self.dishDiameter_in = 20.
		self.dishFocus = 13.42
		self.absDish = 0.1
		self.sigma = 3.1e-3
		self.sigma_in=1.95e-3

		# Source parameters ____________________________________________________________________
		self.num_rays = 100000.

		self.termination_criteria = N.array([0.001])


	def generate_scene(self, geom_mins=None, geom_maxs=None):
		'''
		Generates a set of parameters for a random candidate cooresponding to a predetermined set of geometrical boundary conditions. these parameters sets are stored in the Population class level to be called for ray-traces.
		'''
		new = (geom_mins==None) and (geom_maxs==None)
		N_elems=self.N_elems
		wall_insu=self.wall_insu
		cone_cut_radius=self.cone_cut_radius
		passive_cone=self.passive_cone
		pancake=self.pancake
		elements_per_section=self.elements_per_section

		if new:
			min_rs = self.min_r*N.ones(N_elems+1)
			if cone_cut_radius!=None:
				min_rs[0] = cone_cut_radius
			min_rs[-1] = 0.
			max_rs=self.max_r*N.ones(N_elems+1)
			max_rs[0] = max_rs[0]+wall_insu
			max_rs[-1] = 0.
		else:
			min_rs = geom_mins[0]
			min_xs = geom_mins[1]
			max_rs = geom_maxs[0]
			max_xs = geom_maxs[1]

		# Generate new receiver and optics parameters.
		# 1 - Receiver parameters
		while 1:
			# Aperture
			ap_rad = random.uniform(min_rs[0], max_rs[0])
			# frusta sections
			fru_rads = N.zeros(N_elems-1)
			fru_xs = N.zeros(N_elems-1)
			prev_xs = 0.
			for node in xrange(len(fru_rads)):
				fru_rads[node] = N.around(N.random.uniform(low=min_rs[node+1], high=max_rs[node+1]), decimals=10)
				if new!=True:
					fru_xs[node] = N.around(N.random.uniform(low=prev_xs, high=max_xs[node+1]), decimals=10)
					prev_xs = fru_xs[node]
			if new:
				fru_xs = N.around(N.sort(N.random.uniform(low=0., high=self.max_d, size= N_elems-1)), decimals=10)
			fru_depths = N.add.accumulate(fru_xs)
			# bottom
			if new:
				cone_depth = N.around(random.uniform(-fru_xs[-1], self.max_d-fru_xs[-1]), decimals=10)
			else:
				cone_depth = N.around(random.uniform(-fru_xs[-1]+min_xs[-1], max_xs[-1]-fru_xs[-1]), decimals=10)
			# y = cx+x0
			c = -fru_rads[-1]/cone_depth
			x0 = -c*(fru_xs[-1]+cone_depth)

			# Ensure that the cone does not "eat" the geometry.
			while ((fru_xs*c+x0)>fru_rads).any():
				# bottom
				if new:
					cone_depth = N.around(random.uniform(-fru_xs[-1], self.max_d-fru_xs[-1]), decimals=10)
				else:
					cone_depth = N.around(random.uniform(-fru_xs[-1]+min_xs[-1], -fru_xs[-1]+max_xs[-1]), decimals=10)
				#y = cx+x0	
				c = -fru_rads[-1]/cone_depth
				x0 = -c*(fru_xs[-1]+cone_depth)
			if cone_cut_radius != None:
				fru_rads = N.append(fru_rads, cone_cut_radius)
				fru_depths = N.append(fru_depths,cone_depth-cone_cut_radius/c)
				fru_xs = N.append(fru_xs, fru_xs[-1]+fru_depths[-1])
				cone_depth -= fru_depths[-1] 

				if passive_cone != True:
					# cut the cone:
					cone_depth = 0.
		
			# Tests on geometrical constraints imposed by the cylindrical bounding box:
			if ((N.add.accumulate(N.append(fru_depths,cone_depth))<self.min_d).any()) or ((N.add.accumulate(N.append(fru_depths,cone_depth))>self.max_d).any()):
				continue
			else: 
				break

		if (pancake!=0.) and (pancake!=None):
			aperture_position = -N.sum(fru_depths[:pancake])
		else:
			aperture_position = 0.
		receiver_params = [ap_rad, fru_rads, fru_depths, cone_depth, wall_insu, self.absReceiver, self.emsReceiver, aperture_position, self.envelope_radius, self.envelope_depth, self.specular_receiver]

		# 2 - View factors and areas:
		#	Calculates the view factors matrix, and surface areas of the elemenst of the receiver and associates them to the generated scene.
		VF_scene = TwoNparamcav(*receiver_params)

		#receiver_VF, receiver_areas, receiver_bins_frusta, receiver_bins_cone =0,0,0,0
		receiver_VF, receiver_areas, receiver_bins_frusta, receiver_bins_cone = VF_scene.VF_sim(bins_frusta=N.ones(N_elems-1, dtype=int)*self.elements_per_section, bins_cone=self.elements_per_section, num_rays =100000., precision=0.001)
		del(VF_scene)

		# 3 - Optics parameters:
		#	Generates a set of optical parameters to call for the builder.
		optics_params = [self.dishDiameter, self.dishFocus, self.absDish, self.sigma, self.dishDiameter_in, self.sigma_in]

		return receiver_params, optics_params, receiver_VF, receiver_areas, receiver_bins_frusta, receiver_bins_cone

	def get_geometry(self, params):

		params = params[0]
		
		ap_rad = params[0]
		fru_rads = params[1]
		fru_depths = params[2]
		cone_depth = params[3]

		return N.vstack((N.hstack((ap_rad, fru_rads,0.)),N.add.accumulate(N.hstack((0.,fru_depths,cone_depth)))))

	def build_scene(self, receiver_params, optics_params, receiver_VF, receiver_areas, receiver_bins_frusta, receiver_bins_cone):
		# Builds a raytracing scene from known parameters. These parameters can be declared on the spot of obtained from a randomly generated scene.
		self.Scene = Assembly()
		
		self.Receiver = TwoNparamcav(*receiver_params)
		self.Optics = SG4(*optics_params)

		self.Scene.add_assembly(self.Receiver)
		self.Scene.add_assembly(self.Optics)

		self.Receiver.VF = receiver_VF
		self.Receiver.areas = receiver_areas
		self.Receiver.	bins_frusta = receiver_bins_frusta
		self.Receiver.	bins_cone = receiver_bins_cone

	def generate_source(self, procs=1, flux=1000.):

		# Source parameters
		nrays = self.num_rays/float(procs)
		sourceCenter = N.array([[0,0,2.*self.Optics.dishFocus]]).T # Source center position
		sourceDirection = N.array([0,0,-1.]) # Source normal direction
		sourceRadius = 0.6*self.Optics.dishDiameter # m, Source radius
		sourceAngle = 4.65e-3 # radians, sun rays angular range
		CSR = 0.0225
		G = flux/float(procs)

		#return [nrays, sourceCenter, sourceDirection, sourceRadius, CSR, G]
		return buie_sunshape(nrays, sourceCenter, sourceDirection, sourceRadius, CSR, G)

	def ray_sim(self, params):

		self.build_scene(*params)
		self.Scene.engine = TracerEngine(self.Scene)
		#self.Scene.build_engine()
		#viewer = Renderer(self.Scene.engine)
		#viewer.show_geom()
		results = self.Scene.engine.ray_sim(self.generate_source())

		receiver_bins, self.Receiver_abs, self.Receiver_hits = self.get_receiver().bin_hits()
		optics_hits = self.Scene.get_hits(self.get_optics())
		self.Optics_abs = optics_hits[0]
		self.Optics_hits = optics_hits[1]
		# Blockage and spillage losses:
		# Flux input (just considering useful rays from the source).
		self.flux_input = N.sum(self.Scene.engine.tree._bunds[0].get_energy()[self.Scene.engine.tree._bunds[1].get_parents()])

		# Receiver reflectivity losses:
		# Is considered receiver reflectivity losses all that is lost from the raytrace and not identified as one of the previous losses.
		rec_ref = 0
		spill = 0	
		block = 0

		for i in xrange(1,len(self.Scene.engine.tree._bunds)-1):
			bund_i_rays = self.Scene.engine.tree._bunds[i].get_num_rays()
			bund_i_parents = self.Scene.engine.tree._bunds[i].get_parents()
			bund_i_ener = self.Scene.engine.tree._bunds[i].get_energy()
			bund_im1_ener = self.Scene.engine.tree._bunds[i-1].get_energy()
			bund_ip1_parents = self.Scene.engine.tree._bunds[i+1].get_parents()

			hits = N.zeros(bund_i_rays)
			# hits = 1 means that actual rays hit something and will exist in the next bundle.
			hits[bund_ip1_parents] = 1
			# miss = 1 means that the ray is not hitting anything and will not exist in the next bundle.
			miss = hits != 1
			# disappear means that the ray has hit something and was completely lost afterwards, as a consequence we know it didn't hit an active surface.
			disappeared = bund_i_parents[bund_i_ener < self.Scene.minener]

			if i==1:
				block += N.sum(bund_im1_ener[disappeared])
				spill += N.sum(bund_i_ener[miss])
			elif i==2:
				spill += N.sum(bund_im1_ener[disappeared])
				rec_ref += N.sum(bund_i_ener[miss])
			else:
				rec_ref += N.sum(bund_i_ener[miss])
	
		self.block_losses = block
		self.rec_ref_losses = rec_ref
		self.spill_losses = spill

		return receiver_bins

	def get_receiver(self):
		return self.Scene._assemblies[0]

	def get_optics(self):
		return self.Scene._assemblies[1]
	
	def multi_ray_sim(self, params, procs=8):

		self.Receiver_abs = []
		self.Receiver_hits = []
		self.Optics_abs = []
		self.Optics_hits = []

		self.flux_input = 0.
		self.block_losses = 0.
		self.rec_ref_losses = 0.
		self.spill_losses = 0.

		receiver_bins = []

		self.build_scene(*params)
		self.Scene.engine = TracerEngineMP(self.Scene)

		sources = []
		for i in xrange(procs):
			sources.append(self.generate_source(procs=procs))
		
		results = self.Scene.engine.multi_ray_sim(sources, procs)
		self.Scene = self.Scene.engine._asm
		
		receiver_bins, self.Receiver_abs, self.Receiver_hits = self.get_receiver().bin_hits()
		optics_hits = self.get_optics().get_hits()
		self.Optics_abs = optics_hits[0]
		self.Optics_hits = optics_hits[1]
		# Blockage and spillage losses:
		# Flux input (just considering useful rays from the source).
		self.flux_input = N.sum(self.Scene.engine.tree._bunds[0].get_energy()[self.Scene.engine.tree._bunds[1].get_parents()])

		# Receiver reflectivity losses:
		# Is considered receiver reflectivity losses all that is lost from the raytrace and not identified as one of the previous losses.
		rec_ref = 0
		spill = 0	
		block = 0

		for i in xrange(1,len(self.Scene.engine.tree._bunds)-1):
			bund_i_rays = self.Scene.engine.tree._bunds[i].get_num_rays()
			bund_i_parents = self.Scene.engine.tree._bunds[i].get_parents()
			bund_i_ener = self.Scene.engine.tree._bunds[i].get_energy()
			bund_im1_ener = self.Scene.engine.tree._bunds[i-1].get_energy()
			bund_ip1_parents = self.Scene.engine.tree._bunds[i+1].get_parents()

			hits = N.zeros(bund_i_rays)
			# hits = 1 means that actual rays hit something and will exist in the next bundle.
			hits[bund_ip1_parents] = 1
			# miss = 1 means that the ray is not hitting anything and will not exist in the next bundle.
			miss = hits != 1
			# disappear means that the ray has hit something and was completely lost afterwards, as a consequence we know it didn't hit an active surface.
			disappeared = bund_i_parents[bund_i_ener < self.Scene.engine.minener]

			if i==1:
				block += N.sum(bund_im1_ener[disappeared])
				spill += N.sum(bund_i_ener[miss])
			elif i==2:
				spill += N.sum(bund_im1_ener[disappeared])
				rec_ref += N.sum(bund_i_ener[miss])
			else:
				rec_ref += N.sum(bund_i_ener[miss])
	
		self.block_losses = block
		self.rec_ref_losses = rec_ref
		self.spill_losses = spill

		return receiver_bins

	def get_objectives(self, params, procs=8):
		'''
		Perform the simulation and evaluates the objectives for every candidate.
		'''
		if procs == 1:
			self.Receiver.bin_abs = self.ray_sim(params)
		else:
			self.Receiver.bin_abs = self.multi_ray_sim(params, procs)

		convergence_on_mf = self.Receiver.energy_balance(self.Tamb, self.Trec_in, self.p_in, self.Trec_out, self.tube_diameters_in, self.tube_diameters_out, self.tube_conductivity, self.passive)
		if convergence_on_mf == 'bad_geom':
			return 0. 
		# RESULTS:
		self.Receiver_total_abs = N.sum(self.Receiver.bin_abs)
		#print round(self.Receiver_total_abs), ' W'
		# Optical losses total:
		self.total_optical_losses = self.flux_input-self.Receiver_total_abs
		#print self.Receiver.T_guess
		#print self.Receiver.emissive_losses, N.sum(self.Receiver.Q), self.Receiver.Q

		total_efficiency = (self.Receiver_total_abs-self.Receiver.emissive_losses)/self.flux_input
		#print self.Receiver.T
		#self.Scene.reset_opt()

		return total_efficiency

class Homogen_optim(USASEC_optim):

	def __init__(self):
		USASEC_optim.__init__(self)
		#self.termination_criteria = N.array([0.01, 0.01, 0.01, 0.01])
		self.termination_criteria = N.array([0.001, 0.001, 0.001])
		
	def get_objectives(self, params, procs):

		total_efficiency = USASEC_optim.get_objectives(self, params, procs)

		flux = self.Receiver.bin_abs/self.Receiver.areas[1:]
		self.Receiver.flux = flux
		average_flux = N.sum(self.Receiver.bin_abs)/N.sum(self.Receiver.areas[1:])

		nbins = len(self.Receiver.areas[1:])

		sigma_flux = N.sqrt(N.sum(flux**2.)/nbins-N.sum((flux/nbins)**2.))
		flux_homogeneity = 1./(1.+sigma_flux/average_flux)
		if N.isnan(flux_homogeneity):
			flux_homogeneity = 0.

		Tareas = self.Receiver.T_guess[1:]*self.Receiver.areas[1:]
		sigma_T = N.sqrt(N.sum(Tareas**2.)/N.sum(self.Receiver.areas[1:])-N.sum((Tareas/N.sum(self.Receiver.areas[1:]))**2.))
		average_T = N.sum(Tareas)/N.sum(self.Receiver.areas[1:])
		T_homogeneity = 1./(1.+sigma_T/average_T)

		tube_length = 1./(1.+N.sum(self.Receiver.tube_lengths))

		return N.array([total_efficiency, flux_homogeneity, T_homogeneity, tube_length])
		#return N.array([total_efficiency, T_homogeneity, tube_length])

class Validation_scene():
	'''
	A simple cylindrical receiver canvas to verify the behavior of the optimisation.
	The objectives are:
		Optical efficiency: Ratio of absorbed radiation to total incoming on the aperture of the Dish
		Flux homogeneity: Standard deviation of the fulux on the walls of the receiver
		Surface area: overall surface area of the receiver
	'''
	def __init__(self, rmin=0., rmax=1., lmin=0., lmax=1., N_depth=1., N_ang=1., N_rad=1.):

		self.termination_criteria = N.ones(3)*0.001
		self.starting_geometry_boundaries = [N.array([rmin, lmin]), N.array([rmax,lmax])]
 
		def f_length(rim_angle, rd=0.5):
			rim_angle = rim_angle*N.pi/180.
			f1 = rd/(2.*N.tan(rim_angle))*(1.-N.sqrt(1.+N.tan(rim_angle)**2.))
			f2 = rd/(2.*N.tan(rim_angle))*(1.+N.sqrt(1.+N.tan(rim_angle)**2.))

			if f1>0.:
				focal_length = f1
			else:
				focal_length = f2

			return focal_length

		# Optics parameters ____________________________________________________________________
		self.dishDiameter = 2.
		self.dishFocus = f_length(rim_angle=45., rd=self.dishDiameter/2.)
		self.absDish = 0.1
		self.sigma = None

		# Receiver general parameters:
		self.wall_thickness = 0.01
		self.absReceiver = 0.9

		self.N_depth = N_depth
		self.N_ang= N_ang
		self.N_rad= N_rad

	def generate_scene(self, geom_mins=None, geom_maxs=None):

		new = (geom_mins==None) and (geom_maxs==None)
		if new:
			geom_mins = self.starting_geometry_boundaries[0]
			geom_maxs = self.starting_geometry_boundaries[1]
			
		# Random generator:
		#radius = N.sqrt(geom_mins[0]**2.+N.random.uniform()*(geom_maxs[0]**2.-geom_mins[0]**2.))
		radius = N.random.uniform(low=geom_mins[0], high=geom_maxs[0])
		length = N.random.uniform(low=geom_mins[1], high=geom_maxs[1])

		N_depth = self.N_depth
		N_ang= self.N_ang
		N_rad= self.N_rad

		#Build RTVF assembly:

		cylinder = Surface(FiniteCylinder(diameter=2.*radius, height=length), LambertianReceiver(1), location=N.array([0,0,length/2.]))
		bottom = Surface(RoundPlateGM(Re=radius), LambertianReceiver(1), location=N.array([0,0,length]))
		aperture = AssembledObject(surfs=[Surface(RoundPlateGM(Re=radius), LambertianReceiver(1), location=N.array([0,0,length]))])
		cavity = AssembledObject(surfs=[cylinder, bottom])

		VF_scene = Assembly(objects=[aperture, cavity])
		'''
		Binning scheme array: the result is 3D array: 1st axis is the binned element id, second one is angular min and max bin values, third one is height min and max values.
		structure of the array is:
		axis 0: element
		axis 1: type of data: [0]=angle, [1]=height, [2]=radius
		axis 2: [0]=mini, [1]=maxi
		'''
		binning_scheme = N.zeros((1+N_depth*N_ang+N_rad*N_ang+3, 3,2))

		angles = N.arange(0., 2.*N.pi+2.*N.pi/N_ang, 2.*N.pi/N_ang)
		depths_rec = N.arange(0., length*(1.+1./N_depth), length/N_depth)
		rads_rec = N.arange(0., radius*(1.+1./N_rad), radius/N_rad)

		# Aperture:
		binning_scheme[0] = N.array([[0.,2.*N.pi], [0.,0.], [0.,radius]])
		# Envelope:
		binning_scheme[-3] = N.array([[0.,2.*N.pi], [0.,0.], [radius,radius+self.wall_thickness]])
		binning_scheme[-2] = N.array([[0.,2.*N.pi], [0.,length+self.wall_thickness], [radius+self.wall_thickness,radius+self.wall_thickness]])
		binning_scheme[-1] = N.array([[0.,2.*N.pi], [length+self.wall_thickness,length+self.wall_thickness], [0.,radius+self.wall_thickness]])

		# Cylinder:
		# Radii:
		binning_scheme[1:N_depth*N_ang+1,2,:] = radius
		for d in xrange(N_depth):
			# Angles:
			binning_scheme[1+d*N_ang:1+(d+1)*N_ang,0,0] = angles[:-1]
			binning_scheme[1+d*N_ang:1+(d+1)*N_ang,0,1] = angles[1:]
			# Depths:
			binning_scheme[1+d*N_ang:1+(d+1)*N_ang,1,0] = depths_rec[d]
			binning_scheme[1+d*N_ang:1+(d+1)*N_ang,1,1] = depths_rec[d+1]

		# Flat:
		# Depths:
		binning_scheme[N_depth*N_ang+1:-3,1,:] = length
		for d in xrange(N_rad):
			# Angles:
			binning_scheme[1+N_depth*N_ang+d*N_ang:1+N_depth*N_ang+(d+1.)*N_ang,0,0] = angles[:-1]
			binning_scheme[1+N_depth*N_ang+d*N_ang:1+N_depth*N_ang+(d+1.)*N_ang,0,1] = angles[1:]
			# Radii:
			binning_scheme[1+N_depth*N_ang+d*N_ang:1+N_depth*N_ang+(d+1.)*N_ang,2,0] = rads_rec[d]
			binning_scheme[1+N_depth*N_ang+d*N_ang:1+N_depth*N_ang+(d+1.)*N_ang,2,1] = rads_rec[d+1]

		#Areas
		self.areas = N.zeros(N.shape(binning_scheme)[0])
		for i in xrange(len(self.areas)):
			L = N.sqrt((binning_scheme[i,1,1]-binning_scheme[i,1,0])**2.+((binning_scheme[i,2,1]-binning_scheme[i,2,0])**2.))
			self.areas[i] = (binning_scheme[i,0,1]-binning_scheme[i,0,0])*(binning_scheme[i,2,1]+binning_scheme[i,2,0])/2.*L

		#VF calculations:
		#self.VF = Cav_3D_RTVF(VF_scene, binning_scheme, self.areas, num_rays=10000, precision=0.01, concave=True).VF
		self.binning_scheme = binning_scheme

		return radius, length, self.areas, self.binning_scheme
 
	def generate_source(self, procs, flux=1000., num_rays=100000):

		self.num_rays = num_rays
		# Source parameters
		nrays = num_rays/float(procs)
		sourceCenter = N.array([[0,0,2.*self.dishFocus]]).T # Source center position
		sourceDirection = N.array([0,0,-1.]) # Source normal direction
		sourceRadius = 0.6*self.dishDiameter # m, Source radius
		sourceAngle = 4.65e-3 # radians, sun rays angular range
		CSR = 0.0225
		G = flux/float(procs)

		#return [nrays, sourceCenter, sourceDirection, sourceRadius, CSR, G]
		#return buie_sunshape(nrays, sourceCenter, sourceDirection, sourceRadius, CSR, G)
		return solar_disk_bundle(nrays, sourceCenter, sourceDirection, sourceRadius, sourceAngle, G)

	def build_scene(self, radius, length, areas, binning_scheme):
	
		cylinder = Surface(FiniteCylinder(diameter=2.*radius, height=length), LambertianReceiver(self.absReceiver), location=N.array([0,0,length/2.]))
		bottom = Surface(RoundPlateGM(Re=radius), LambertianReceiver(self.absReceiver), location=N.array([0,0,length]))
		envelope_front = Surface(RoundPlateGM(Re=radius+self.wall_thickness, Ri=radius), LambertianReceiver(1.))
		envelope_side = Surface(FiniteCylinder(diameter=2.*(radius+self.wall_thickness), height=length+self.wall_thickness), LambertianReceiver(1.), location=N.array([0,0,(length+self.wall_thickness)/2.]))
		envelope_back = Surface(RoundPlateGM(Re=radius+self.wall_thickness), LambertianReceiver(1.), location=N.array([0,0,length+self.wall_thickness]))

		self.Receiver = AssembledObject(surfs=[cylinder,bottom])
		self.Envelope = AssembledObject(surfs=[envelope_front, envelope_side, envelope_back])
		#self.Optics = 	AssembledObject(surfs=[Surface(ParabolicDishGM(self.dishDiameter, self.dishFocus), RealReflectiveReceiver(self.absDish, self.sigma))], transform=translate(z=-self.dishFocus))
		self.Optics = 	AssembledObject(surfs=[Surface(ParabolicDishGM(self.dishDiameter, self.dishFocus), ReflectiveReceiver(self.absDish))], transform=translate(z=-self.dishFocus))

		self.Scene = Assembly(objects=[self.Receiver, self.Envelope, self.Optics])

		self.areas = areas # Necessary because only the builder will be called for every get_objectives, not the generator
		self.binning_scheme = binning_scheme

	def get_geometry(self, params):
		return N.vstack(params[:2])

	def get_objectives(self,params,procs):

		sources = []
		for i in xrange(procs):
			sources.append(self.generate_source(procs=procs))
		self.build_scene(*params)
		height = params[1]

		engine = TracerEngineMP(self.Scene)
		self.Scene = engine.multi_ray_sim(sources, procs)

		# Occasional debugging rendering:
		#view = Renderer(engine)
		#view.show_geom()
		#view.show_rays()

		total_input = N.sum(engine.tree._bunds[0].get_energy()[engine.tree._bunds[1].get_parents()])

		receiver_surfs = self.Scene.get_objects()[0].get_surfaces()

		binning_local = self.binning_scheme[1:-3] # Here the aperture is out.
		
		self.bin_abs = N.zeros(N.shape(binning_local)[0])

		for s in xrange(len(receiver_surfs)):
			abso, hits = receiver_surfs[s].get_optics_manager().get_all_hits() # we only do the absorber here but the envelope can be added if needed.
			#print N.shape(abso)
			#print N.sum(abso)
			#print N.shape(hits)

			if len(abso):
				for i in xrange(N.shape(binning_local)[0]):
					ahr = binning_local[i]

					ang0 = ahr[0,0]
					ang1 = ahr[0,1]

					h0 = ahr[1,0]
					h1 = ahr[1,1]

					r0 = ahr[2,0]
					r1 = ahr[2,1]

					angles_absorber = N.arctan2(hits[1], hits[0]) # returns quadran matched in the form opf -pi<x<pi
					angles_absorber[angles_absorber<0.] = 2.*N.pi+angles_absorber[angles_absorber<0.]	
					absorber_in_ang = N.logical_and(angles_absorber>=ang0, angles_absorber<ang1)

					if h0==h1:
						absorber_in_h = (N.round(hits[2], decimals=9) == N.round(h1, decimals=9))
					else:
						absorber_in_h = N.logical_and(hits[2]>=h0, hits[2]<h1)

					rads_absorber = N.sqrt(hits[0]**2.+hits[1]**2.)
					if r0==r1:
						absorber_in_r = (N.round(rads_absorber,decimals=9) == N.round(r1, decimals=9))
					else:
						absorber_in_r = N.logical_and(rads_absorber>=r0, rads_absorber<r1)

					abs_in_bin = absorber_in_h*absorber_in_ang*absorber_in_r
					abs_abs = N.sum(abso[abs_in_bin])

					abso = abso[~abs_in_bin]
					hits = hits[:,~abs_in_bin]
				
					self.bin_abs[i] += abs_abs

		print 'total input :', total_input
		print 'total abs :', N.sum(self.bin_abs)

		optical_efficiency = N.sum(self.bin_abs)/total_input

		#view = Renderer(engine)
		#view.show_geom()
		#view.show_rays()

		flux = self.bin_abs/self.areas[1:-3]
		average_flux = N.sum(flux)/len(flux)
		sigma_flux = N.sqrt(N.sum(flux**2.)/len(flux)-average_flux**2.)
		flux_homogeneity = 1./(1.+sigma_flux/average_flux)
		if N.isnan(flux_homogeneity):
			flux_homogeneity = 0.

		total_area = 1./(1.+N.sum(self.areas[1:-3]))

		return N.array([optical_efficiency, flux_homogeneity, total_area])

