#!/usr/bin/env python
"""Provides a gridded sense of which map regions are empty or not.

An occupancy grid takes the map boundaries, creates a grid (with
arbitrary cell size) and gives each cell a value: 0 for unoccupied
cells, 1 for occupied cells, and None for cells that have not yet
been explored.

Note
----
    The Bernoulli distribution (0 or 1) per cell is ideal, and a true
    occupancy grid would include a continuous probability per cell of
    its likelihood of occupancy. This will likely be added in future
    versions.

    The occupancy grid currently considers only static objects. This
    will also likely change in the future versions.

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Stable"

import logging

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Point

# from cohrint_core.map_tools.layer import Layer


class OccupancyLayer(object):
    """Generate an occupancy grid from static map objects.

    Gridded occupancy layer for the map, translating euclidean coordinates
    to grid cells. Each cell has a probability of occupancy from 0 to 1.

    .. image:: img/classes_.png

    Parameters
    ----------
    cell_size : float, optional
        The side length for each square cell in the occupancy grid in [m].
        Defaults to 0.05.

    """
    def __init__(self, feasible_layer, cell_size=1.0, bounds=[-5, -5, 5, 5]):
        self.bounds = bounds
        self.feasible_layer = feasible_layer
        self.cell_size = cell_size

        self.x_coords = np.arange(self.bounds[0], self.bounds[2], cell_size)
        self.y_coords = np.arange(self.bounds[1], self.bounds[3], cell_size)

        self.occupancy_grid = []
        for i, y in enumerate(self.y_coords):
            self.occupancy_grid.append([])
            for j, x in enumerate(self.x_coords):
                pt = Point([x, y])
                if self.feasible_layer.pose_region.contains(pt):
                    self.occupancy_grid[i].append(0)
                else:
                    self.occupancy_grid[i].append(1)



    #     self.cell_size = cell_size

    #     self.grid = []
    #     x, y = self.bounds[0], self.bounds[1]
    #     c = self.cell_size

    #     # <>TODO: @Matt Redo grid cell generation (it's too slow!)
    #     # Create cells with grid centered on (0,0)
    #     while x + c <= self.bounds[2]:
    #         while y + c <= self.bounds[3]:
    #             cell = box(x, y, x + c, y + c)
    #             self.grid.append(cell)
    #             y = y + c
    #         x = x + c
    #         y = self.bounds[1]

    #     self.n_cells = len(self.grid)

    #     self.grid_occupancy = 0.5 * np.ones((self.n_cells, 1), dtype=np.int)

    # def add_obj(self, map_obj):
    #     """Fill the cells for a given map_obj.

    #     Parameters
    #     ----------
    #     map_obj : MapObj
    #         The object to be added.
    #     """
    #     for i, cell in enumerate(self.grid):
    #         if map_obj.shape.intersects(cell):
    #             self.grid_occupancy[i] = 1

    # def rem_obj(self, map_obj):
    #     """Empty the cells for a given map_obj.

    #     Parameters
    #     ----------
    #     map_obj : MapObj
    #         The object to be removed.
    #     """
    #     for i, cell in enumerate(self.grid):
    #         if map_obj.shape.intersects(cell):
    #             self.grid_occupancy[i] = 0

    # def occupancy_from_shape_layer(self, shape_layer):
    #     """Create an occupancy grid from an entire shape layer.

    #     Parameters
    #     ----------
    #     shape_layer : ShapeLayer, optional
    #         The shape layer from which to generate the feasible regions. If
    #         no layer is provided, the entire map is deemed feasible.

    #     Note
    #     ----
    #     Not yet implemented.
    #     """
    #     pass

    # def plot(self):
    #     """Plot the occupancy grid as a pseudo colormesh.

    #     Returns
    #     -------
    #     QuadMesh
    #         The scatter pseudo colormesh data.

    #     """
    #     xsize = self.bounds[2] - self.bounds[0]
    #     ysize = self.bounds[3] - self.bounds[1]
    #     grid = self.grid_occupancy.reshape(xsize / self.cell_size,
    #                                        ysize / self.cell_size)
    #     X, Y = np.mgrid[0:grid.shape[0]:1, 0:grid.shape[1]:1]
    #     X, Y = (X * self.cell_size, Y * self.cell_size)
    #     p = plt.pcolormesh(X, Y, grid, cmap=cm.Greys)
    #     # <>TODO: add in cell borders!
    #     return p
