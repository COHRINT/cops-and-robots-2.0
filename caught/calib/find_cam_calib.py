"""
    Calculates the calibation equation of best fit
    for a kinect camera using the .csv file
"""


import csv
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

with open('dist_pixel_calib.csv', 'rb') as filename:
    reader = csv.reader(filename)
    # rand_array = np.empty((0,2))
    dist_list = []
    pixel_list = []
    for row in filename:
        (dist, pixel) = [x.strip() for x in row.split(",")]
        dist_list.append(float(dist))
        pixel_list.append(float(pixel))

    print(dist_list)
    print(pixel_list)

distances = np.arange(0,1.1,0.1).tolist()

coefs = []
coefs = poly.polyfit(dist_list, pixel_list, 2)
print(coefs)

jeremy = poly.polyval(distances, coefs)
print(jeremy)
plt.plot(dist_list, pixel_list, 'bo')
plt.plot(distances, jeremy)
plt.show()
