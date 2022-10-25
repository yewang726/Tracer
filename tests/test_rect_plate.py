# Tests for the rectangular plate geometry manager.

import unittest
import numpy as N

from tracer.flat_surface import RectPlateGM, ExtrudedRectPlateGM
from tracer.ray_bundle import RayBundle
from tracer.surface import Surface
import tracer.optics_callables as opt

class TestRectPlateGM(unittest.TestCase):
	def test_value_error(self):
		"""Can't create a negative rect-plate"""
		self.assertRaises(ValueError, RectPlateGM, -1, 7)
		self.assertRaises(ValueError, RectPlateGM, 1, -7)
	
	def test_selection(self):
		pos = N.zeros((3,4))
		pos[0] = N.r_[0, 0.5, 2, -2]
		pos[2] = 1.
		dir = N.tile(N.c_[[0,0,-1]], (1,4))
		bund = RayBundle(pos, dir)
		
		surf = Surface(RectPlateGM(1, 0.25), opt.perfect_mirror)
		misses = N.isinf(surf.register_incoming(bund))
		
		N.testing.assert_array_equal(misses, N.r_[False, False, True, True])
	
	def test_mesh(self):
		"""Correct mesh for rect-plate"""
		r = RectPlateGM(5, 6)
		res = 10
		x, y, z = r.mesh(res)
		
		rx = N.linspace(-2.5, 2.5, res+1)
		ry = N.linspace(-3, 3, res+1)
		cx, cy = N.broadcast_arrays(rx[:,None], ry)		

		N.testing.assert_array_equal(x, cx)
		N.testing.assert_array_equal(y, cy)
		N.testing.assert_array_equal(z, N.zeros_like(x))

class TestExtrudedRectPlateGM(unittest.TestCase):
	
	def test_selection(self):
		pos = N.zeros((3,4))
		pos[0] = N.r_[0, 0.05, 0.2, -0.3]
		pos[2] = 1.
		dir = N.tile(N.c_[[0,0,-1]], (1,4))
		bund = RayBundle(pos, dir)
		
		surf = Surface(ExtrudedRectPlateGM(width=1, height=1, extr_center=N.vstack([0.2,0.1]), extr_width=0.2, extr_height=0.4), opt.perfect_mirror)
		misses = N.isinf(surf.register_incoming(bund))

		N.testing.assert_array_equal(misses, N.r_[False, False, True, False])

