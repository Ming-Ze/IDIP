import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Registration
from VTK_func import *

cube = np.zeros((128, 128, 128))
cube.fill(0)
#z, x, y = cube.shape

index = 128

cube[0,:,:] = index
cube[127,:,:] = index
cube[:,0,:] = index
cube[:,127,:] = index
cube[:,:,0] = index
cube[:,:,127] = index

cube[32, 32:96, 32:96] = index
cube[96, 32:96, 32:96] = index
cube[32:96, 32, 32:96] = index
cube[32:96, 96, 32:96] = index
cube[32:96, 32:96, 32] = index
cube[32:96, 32:96, 96] = index

alpha = 0.5
colors = np.empty([128, 128, 128] + [4], dtype=np.float32) 
colors[:] = [1, 0, 0, alpha]  # red

print(cube)
#print(x, y, z)
#plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')

#plt.axis('off')
#plt.show()

np.save(f'Dicom_new\\result\\cube', cube)

Registration.registration(cube, cube)