import Registration
from VTK_func import *
import numpy as np

def main():
	np_lung = np.load("C:\\Users\\王銘澤\\桌面\\OpenCV\\Dicom_new\\result\\1_mapped.npy").astype(np.uint8)
	print(np_lung)
	z, y, x = np_lung.shape
	print(x, y, z)
	
	ren = vtk.vtkRenderer()
	# lung_nodule_bbox(ren, np_lung, nodules, oribox, bbox_array, show_lung=True, show_bbox=False, show_nodule=True)
	lung_nodule_bbox(ren, np_lung, "LightCoral", show_lung=True, show_bbox=False, show_nodule=False)
	ren.SetBackground(1.0, 1.0, 1.0)

	renWin = vtk.vtkRenderWindow()
	renWin.AddRenderer(ren)
	renWin.SetSize(500, 600)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	# worldPointPicker = vtk.vtkWorldPointPicker()
	# iren.SetPicker(worldPointPicker) # new

	style = MouseInteractorActor()
	style.SetDefaultRenderer(ren)
	iren.SetInteractorStyle(style)

	Camera = ren.GetActiveCamera()
	Camera.SetPosition(0, 1, 0)  # Set/Get the position of the camera in world coordinates.
	Camera.SetViewUp(0, 0, -1)  # Set/Get the view up direction for the camera.
	ren.ResetCamera()


	# Render the scene
	renWin.Render()

	iren.Initialize()
	iren.Start()
	

if __name__ == "__main__":
	#for i in range(1,7):
	#	Registration.dicom_save_as_np(i)
	"""
	img1 = np.load(f"Dicom_new\\result\\1_mapped.npy")
	img2 = np.load(f"Dicom_new\\result\\2_mapped.npy")
	img3 = np.load(f"Dicom_new\\result\\3_mapped.npy")
	img4 = np.load(f"Dicom_new\\result\\4_mapped.npy")
	img5 = np.load(f"Dicom_new\\result\\5_mapped.npy")
	img6 = np.load(f"Dicom_new\\result\\6_mapped.npy")
	"""
	#Registration.registration(1, 2)
	#Registration.registration(3, 4)
	#Registration.registration(5, 6)	

	main()