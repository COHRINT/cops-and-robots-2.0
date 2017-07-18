#!/usr/bin/env python

""" Summary:
	Creates a map object from an inputed 'map.yaml' file (in models dir)
		with softmax LIKELIHOODs
	Map includes:
		1) General info: name, bounds.max_x_y, bounds.min_x_y, origin
		2) Object hash: 'self.objects', each member is a Map_Object
		3) Rooms : self.rooms['room_name']['lower_l' OR 'upper_r' OR 'likelihood']
			access the room's lower left coordinate and upper right coord
	Map_Object includes:
		name, color, centroid[x, y], major axis, minor axis,
		orientation from the object's major axis to the map's positive x axis,
		shape (available shapes: oval and rectangle),
		softmax likelihood
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Luke Babier", "Ian Loefgren", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0" # Likelihoods added
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

import yaml
import os 			# path capabilities
from softmaxModels import *

class Map(object):
	""" Map Object from map.yaml file (located in models dir)

	Map includes:
		1) General info: self.name (str),
						self.size[max_x, max_y] (float list),
						self.origin[x, y] (float list)
		2) Object hash: 'self.objects', each member is a Map_Object

	Parameters
	----------
	yaml_file : map1.yaml, map2.yaml, etc

	"""

	def __init__(self, yaml_file):

		# load yaml file as a dictionary
		# cfg = self._find_yaml(yaml_file)
		cfg = yaml.load(open('../models/' + yaml_file, 'r'));

		if cfg is not None:

			# Get map's general info
			self.name = cfg['info']['name']
			#self.map_bounds.max_x_y = [cfg['info']['bounds']['max_x'], cfg['info']['bounds']['max_y']]
			#self.map_bounds.min_x_y = [cfg['info']['bounds']['min_x'], cfg['info']['bounds']['min_y']]
			self.origin = [cfg['info']['origin']['x_coord'], cfg['info']['origin']['y_coord']]

			# Add room boundaries to the map
			self.rooms = {}
			lower_l = list()
			upper_r = list()
			for room in cfg['info']['rooms']:
				lower_l = (cfg['info']['rooms'][room]['min_x'], cfg['info']['rooms'][room]['min_y'])
				upper_r = (cfg['info']['rooms'][room]['max_x'], cfg['info']['rooms'][room]['max_y'])
				self.rooms[room] = {}
				self.rooms[room]['lower_l'] = lower_l
				self.rooms[room]['upper_r'] = upper_r
				length = upper_r[0] - lower_l[0]
				width = upper_r[1] - lower_l[1]
				cent = [lower_l[0] + length/2,lower_l[1]+width/2]
				self.rooms[room]['softmax'] = Softmax()
				self.rooms[room]['softmax'].buildOrientedRecModel(cent, 0.0, length, width)

			# Store map's objects in self.objects hash
			self.softmax = Softmax()
			self.objects = {}
			for item in cfg:
				if item != 'info':	# if not general info => object on map
					map_obj = Map_Object(cfg[item]['name'],
										cfg[item]['color'],
										[cfg[item]['centroid_x'], cfg[item]['centroid_y']],
										cfg[item]['length'],
										cfg[item]['width'],
										cfg[item]['orientation'],
										cfg[item]['shape']
										)
					self.objects[map_obj.name] = map_obj


	# Searches the yaml_dir for the given yaml_file
	# Returns a python dictionary if successful
	# Returns 'None' for failure
	def _find_yaml(self, yaml_file):
		yaml_dir = 'models'

		try:
			# navigate to yaml_dir
			cfg_file = os.path.dirname(__file__) \
				+ '/../' + yaml_dir + '/' + yaml_file
			# return dictionary of yaml file
			with open(cfg_file, 'r') as file:
				 return yaml.load(file)
		except IOError as ioerr:
			print str(ioerr)
			return None

class Map_Object(object):
	"""
	Objects like chairs, bookcases, etc to be included in the map object
	-Derived from a map.yaml file (in models dir)

	Map_Object includes:
		name (str), color (str), centroid[x, y] (float list), major axis (float),
		minor axis (float),
		orientation from the object's major axis to the map's positive x axis (float)
		shape (str) (available shapes: oval and rectangle)
		softmax likelihood

	Parameters
	----------
	name: str
		Name of obj
	color: str
		Color of obj
	centroid : 2x1 list
		Centroid location [x, y] [m]
	length: float
		x axis length of obj [m] (along direction object is facing)
	width: float
		y axis width of obj [m] (normal to direction object is facing)
	orient : float
		Radians between obj's major axis and the map's pos-x axis
	shape : str
		Values accepted: 'rectangle' or 'oval'
	"""
	def __init__(self,
				name='wall',
				color='darkblue',
				centroid=[0.0,0.0],
				length = 0.0,
				width = 0.0,
				orient=0.0,
				shape = 'rectangle'
				):
		self.name = name
		self.color = color
		self.centroid = centroid
		self.length = length
		self.width = width
		self.orient = orient

		self._pick_shape(shape)

		# create the objects likelihood
		self.softmax = Softmax()
		self.get_likelihood()

	def get_likelihood(self):
		"""
		Create and store corresponding likelihood.
		Approximate all shapes as rectangles
		"""
		self.softmax.buildOrientedRecModel(self.centroid,
			self.orient, self.length, self.width)

	# Selects the shape of the obj
	# Default = 'rectangle' --- 'oval' also accepted
	def _pick_shape(self, shape):

		if shape == 'oval':
			self.shape = 'oval'
		else:
			self.shape = 'rectangle'

def test_map_obj():
	map1 = Map('map2.yaml')

	if hasattr(map1, 'name'): # check if init was successful
		print map1.name
		print map1.objects['dining table'].color
		print map1.rooms['dining room']['lower_l']
		print map1.rooms['kitchen']['upper_r']

	else:
		print 'fail'

def test_likelihood():
	map2 = Map('map2.yaml')
	if hasattr(map2, 'name'):
		for obj in map2.objects:
			print obj
		print("Dining table:")
		print (map2.objects['dining table'].softmax.weights)
		print (map2.objects['dining table'].softmax.bias)
		print (map2.objects['dining table'].softmax.size)
		print("Mars Poster:")
		print(map2.objects['mars poster'].softmax.weights)
		print("Dining Room: ")
		print(map2.rooms['dining room']['softmax'].weights)
	else:
		print("Failed to initialize Map Object.")
		raise

if __name__ == "__main__":
	#test_map_obj()
	test_likelihood()
