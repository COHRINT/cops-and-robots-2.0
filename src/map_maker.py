#!/usr/bin/env python
import yaml
import os 			# path capabilities

""" Summary:
	Creates a map object from an inputed 'map.yaml' file (in models dir)
	Map includes:
		1) General info: name, bounds.max_x_y, bounds.min_x_y, origin
		2) Object hash: 'self.objects', each member is a Map_Object
	Map_Object includes:
		name, color, centroid[x, y], major axis, minor axis,
		orientation from the object's major axis to the map's positive x axis
		shape (available shapes: oval and rectangle)
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Luke Babier", "Ian Loefgren", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"


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
		#cfg = self._find_yaml(yaml_file)
		cfg = yaml.load(open('../models/map1.yaml', 'r'));

		if cfg is not None:

			# Get map's general info
			self.name = cfg['info']['name']
			self.bounds.max_x_y = [cfg['info']['bounds']['max_x'], cfg['info']['bounds']['max_y']]
			self.bounds.min_x_y = [cfg['info']['bounds']['min_x'], cfg['info']['bounds']['min_y']]
			self.origin = [cfg['info']['origin']['x_coord'], cfg['info']['origin']['y_coord']]

			# Store map's objects in self.objects hash
			self.objects = {}
			for item in cfg:
				if item != 'info':	# if not general info => object on map
					map_obj = Map_Object(cfg[item]['name'],
										cfg[item]['color'],
										[cfg[item]['centroid_x'], cfg[item]['centroid_y']],
										cfg[item]['maj_ax'],
										cfg[item]['min_ax'],
										cfg[item]['orientation'],
										cfg[item]['shape'])
					self.objects[map_obj.name] = map_obj


	# Searches the yaml_dir for the given yaml_file
	# Returns a python dictionary if successful
	# Returns 'None' for failure
	def _find_yaml(self, yaml_file):
		yaml_dir = 'models'

		try:
			# navigate to yaml_dir
			cfg_file = os.path.dirname(__file__) \
				+ '../' + yaml_dir + '/' + yaml_file
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

	Parameters
	----------
	name: str
		Name of obj
	color: str
		Color of obj
	centroid_pos : list
		Centroid location [x, y] [m]
	x_ax_len: float
		x axis length of obj [m] (before orientation adjustment)
	min_ax_len: float
		y axis length of obj [m] (before orientation adjustment)
	orient : float
		Radians between obj's major axis and the map's pos-x axis
	shape : str
		Values accepted: 'rectangle' or 'oval'
	"""
	def __init__(self,
				name='wall',
				color='darkblue',
				centroid_pos=[0.0,0.0],
				x_len = 0.0,
				y_len = 0.0,
				orient=0.0,
				shape = 'rectangle'
				):
		self.name = name
		self.color = color
		self.centroid = centroid_pos
		self.x_len = maj_ax_len
		self.y_len = min_ax_len
		self.orient = orient

		self._pick_shape(shape)

	# Selects the shape of the obj
	# Default = 'rectangle' --- 'oval' also accepted
	def _pick_shape(self, shape):

		if shape == 'oval':
			self.shape = 'oval'
		else:
			self.shape = 'rectangle'

def test_map_obj():
	map = Map('map2.yaml')

	if hasattr(map, 'name'): # check if init was successful
		print map.name
		print map.objects['table'].color
	else:
		print 'fail'

if __name__ == "__main__":
	test_map_obj()
