import SimpleITK as sitk
import os
import numpy as np
import gui
import registration_gui as rgui

def map_img(img, window=np.array([-1200, 600]), dtype="uint8"):
    '''
    img: image
    window: original floor and ceiling of the image to be preserved
    dtype: dtype of mapped image
    '''
    info = np.iinfo(dtype)
    img_normalized = (img - min(window)) / (max(window) - min(window))
    img_truncated = np.maximum(np.minimum(img_normalized, 1), 0)
    img_mapped = (img_truncated * (info.max - info.min) + info.min).astype(dtype)
    return img_mapped

def read_dicom(id):
    dicom_path = f'Dicom_new\\dicom\\Id000{id}'
    reader = sitk.ImageSeriesReader()
    dicom_names = tuple()
    series_ids = reader.GetGDCMSeriesIDs(dicom_path)  # Get all the seriesIDs from a DICOM data set.
    for series_id in series_ids:
        id_names_dcm = reader.GetGDCMSeriesFileNames(dicom_path, seriesID=series_id)
        dicom_names = id_names_dcm + dicom_names
    reader.SetFileNames(dicom_names)
    original_image = reader.Execute()

    size = original_image.GetSize()
    spacing = original_image.GetSpacing()
    direction = original_image.GetDirection()
    origin = original_image.GetOrigin()

    np_original_image = sitk.GetArrayFromImage(original_image)

    mask = np.load(f'Dicom_new\\mask\\Id000{id}.npy')
    image = np_original_image * mask
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)
    result = sitk.Cast(sitk_image, sitk.sitkFloat32)
    print(f'read dicom Id{id} success')

    return result


def registration(id1, id2):
    fixed_image = read_dicom(id1)  # image transformed from fixed_image's coordinate to moving_image's coordinate
    print(fixed_image.GetSize())
    moving_image = read_dicom(id2)
    print(moving_image.GetSize())
    #fixed_image = id1
    #moving_image = id2

    ct_window_level = [835, 162]
    mr_window_level = [1036, 520]

    gui.MultiImageDisplay(image_list=[fixed_image, moving_image],
                          title_list=['fixed', 'moving'], figure_size=(8, 4), window_level_list=[ct_window_level, mr_window_level])
    # Initial Transform
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, convergenceMinimumValue=1e-4, numberOfIterations=100, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))

    # the final metric we are going to use
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(final_transform)
    print(final_transform.TransformPoint((161, 330, 94)))  # (x,y,z)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    sitk.WriteTransform(final_transform, os.path.join("C:\\Users\\王銘澤\\桌面\\OpenCV\\Dicom_new\\result",
                                                      f"{id1}_{id2}_final_transform.tfm"))
    print(f"{id1}_{id2}_final transform value is saved")

    # registration result
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    print(moving_resampled.GetSize())
    npImage = sitk.GetArrayFromImage(moving_resampled)
    np.save(f"Dicom_new\\result\\{id1}_{id2}_result", npImage)
    print(f"{id1}_{id2}_result numpy file is saved")


def dicom2np(id):
    dicom_image = read_dicom(id)
    npImage = sitk.GetArrayFromImage(dicom_image)
    return npImage


def dicom_save_as_np(id):
    image = dicom2np(id)
    #np.save(f"Dicom_new\\result\\{id}_npImage", image)
    image_mapped = map_img(image)
    np.save(f"Dicom_new\\result\\{id}_mapped", image_mapped)