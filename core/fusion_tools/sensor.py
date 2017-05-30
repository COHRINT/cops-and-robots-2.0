#!/usr/bin/env python
"""Provides an base class for various sensor types.

Since many sensors share parameters and functions, the ``sensor``
module defines these in one place, allowing all sensors to use it as
a superclass.

Note
----
    Only cop robots have sensors (for now). Robbers may get hardware
    upgreades in future versions, in which case this would be owned by
    the ``robot`` module instead of the ``cop`` module.

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


class Sensor(object):
    """Base class for all sensor models.

    .. image:: img/classes_Sensor.png

    Parameters
    ----------
    update_rate : float
        Frequency of sensor updates in Hz. `None` for intermittant updates.

        Note: Not yet implemented.
    has_physical_dimensions : bool
        Whether or not the sensor can be considered a physical sensor (i.e. a
        camera).
    detection_chance : float
        A probability value between 0 and 1 denoting P(detect|x), when x is in
        view of the sensor.

    """
    def __init__(self, update_rate, has_physical_dimensions, detection_chance=0):
        super(Sensor, self).__init__()

        # Define simlated sensor parameters
        self.update_rate = update_rate  # [hz]
        self.has_physical_dimensions = has_physical_dimensions
        self.detection_chance = detection_chance  # P(detect|x), x is in view
