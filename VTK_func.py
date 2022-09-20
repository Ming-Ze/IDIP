import numpy as np
import vtk

# step : 1.DataImporter 2.(MarchingCube) 3.Mapper 
#        4.Actor 5.Render 6.Renderer Window 

# turn seg_3d_re_bits form 1-D array to 3-D array in clean_lung shape
def unpack_seg_3d_re_bits(seg_3d_re_bits, oribox):
    # print(len(seg_3d_re_bits))
    z1_crop, z2_crop = oribox[0]
    y1_crop, y2_crop = oribox[1]
    x1_crop, x2_crop = oribox[2]
    v = z2_crop - z1_crop
    h = y2_crop - y1_crop
    w = x2_crop - x1_crop
    # print("v, h, w")
    # print(v, h, w)
    # print(seg_3d_re_bits.shape)
    seg_3d_re = np.unpackbits(seg_3d_re_bits)[:v*h*w].reshape(v, h, w)
    # print(seg_3d_re.shape)
    seg_3d_re = seg_3d_re[::-1, :, :]
    # use "astype(np.uint8)*255" in order to draw the nodule ( 0->0 , 1->255 ) 
    return seg_3d_re.astype(np.uint8)*255

#----------------------------------------------------------------------------
# add numpy in dataImporter
def make_DataImporter(numpy_array):
    # vtk_nodule = numpy_support.numpy_to_vtk(num_array=nodule.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    dataImporter = vtk.vtkImageImport()
    string = numpy_array.tostring()
    dataImporter.CopyImportVoidPointer(string, len(string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    x, y, z = numpy_array.shape
    # print(f'x,y,z = {x,y,z}')
    dataImporter.SetDataExtent(0, z-1, 0, y-1, 0, x-1)
    dataImporter.SetWholeExtent(0, z-1, 0, y-1, 0, x-1)
    return dataImporter

#----------------------------------------------------------------------------
# marching cube in order to make the nodule contour
def marching_cube_nodule(dataImporter):
    extractor_nodule = vtk.vtkMarchingCubes()
    extractor_nodule.SetInputConnection(dataImporter.GetOutputPort())
    extractor_nodule.ComputeNormalsOn()
    extractor_nodule.ComputeScalarsOn()
    extractor_nodule.SetValue(0, 10) # threshold
    extractor_nodule.Update()   
    return extractor_nodule

# marching cube in order to make the lung contour
def marching_cube_lung(dataImporter):
    extractor = vtk.vtkMarchingCubes()
    extractor.SetInputConnection(dataImporter.GetOutputPort())
    extractor.ComputeNormalsOn()
    extractor.ComputeScalarsOn()
    extractor.SetValue(0, 127) # threshold
    extractor.Update()  
    return extractor

#----------------------------------------------------------------------------
# Create Mapper
def create_mapper(extractor):
    polydataMapper = vtk.vtkPolyDataMapper() 
    polydataMapper.SetInputConnection(extractor.GetOutputPort()) 
    polydataMapper.ScalarVisibilityOff()
    return polydataMapper

#----------------------------------------------------------------------------
# Create Nodule Actor
def create_nodule(polydataMapper):
    actor_nodule = vtk.vtkActor()
    actor_nodule.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("DarkTurquoise")) #結核顏色DarkTurquoise
    actor_nodule.SetMapper(polydataMapper) 
    actor_nodule.PickableOn() # get info. by pick actor 
    return actor_nodule

# Create Lung Actor
def create_lung(polydataMapper, color):
    actor = vtk.vtkActor()
    actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(color)) #肺部顯示顏色LightCoral
    actor.GetProperty().SetOpacity(0.2) #0.3
    actor.SetMapper(polydataMapper)  
    actor.PickableOff() 
    return actor
#----------------------------------------------------------------------------
def draw_lung(nparray, color):
    # make DataImporter
    dataImporter = make_DataImporter(nparray)
    # marching cube in order to make the lung
    extractor = marching_cube_lung(dataImporter)
    # Create mapper
    polydataMapper = create_mapper(extractor)
    # Create lung actor
    actor_lung = create_lung(polydataMapper, color)
    return actor_lung

def draw_nodule(seg_3d_re_bits, oribox):
    nodule = (unpack_seg_3d_re_bits(seg_3d_re_bits, oribox))
    # make DataImporter
    dataImporterNodule = make_DataImporter(nodule)
    # marching cube in order to make the nodule
    extractor_nodule = marching_cube_nodule(dataImporterNodule)
    # Create mapper
    polydataMapper_nodule = create_mapper(extractor_nodule)
    # Create lung actor
    actor_nodule = create_nodule(polydataMapper_nodule)

    return actor_nodule

def tmp_draw_bbox(bbox):
    z, y, x, d = bbox[:4]
    cube = vtk.vtkCubeSource()
    cube.SetCenter(x, y, z)
    cube.SetXLength(d)
    cube.SetYLength(d)
    cube.SetZLength(d)
    cube.Update()

    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputConnection(cube.GetOutputPort())
    featureEdges.BoundaryEdgesOn()
    featureEdges.FeatureEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    # featureEdges.ColoringOn()
    featureEdges.ColoringOff()
    # featureEdges.GetColoring()
    featureEdges.Update()

    cm = vtk.vtkPolyDataMapper()
    cm.SetInputConnection(featureEdges.GetOutputPort())
    # cm.SetScalarModeToUseCellData()

    # ca = vtk.vtkAnnotatedCubeActor()
    # ca.GetCubeProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Blue"))
    
    ca = vtk.vtkActor()
    ca.PickableOff() 
    ca.SetMapper(cm)
    ca.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Violet"))
    # ca.GetProperty().EdgeVisibilityOff()

    # ca.GetProperty().SetEdgeColor(vtk.vtkNamedColors().GetColor3d("Aqua"))
    # ca.GetProperty().SetLineWidth(1.5)

    return ca

def draw_bbox(bbox, color="Blue"):
    z, y, x, d = bbox[:4]

    # Create the polydata where we will store all the geometric data
    linesPolyData = vtk.vtkPolyData()

    # Create 8 points
    p0 = [x - d/2, y + d/2, z + d/2]
    p1 = [x + d/2, y + d/2, z + d/2]
    p2 = [x - d/2, y - d/2, z + d/2]
    p3 = [x + d/2, y - d/2, z + d/2]
    p4 = [x - d/2, y + d/2, z - d/2]
    p5 = [x + d/2, y + d/2, z - d/2]
    p6 = [x - d/2, y - d/2, z - d/2]
    p7 = [x + d/2, y - d/2, z - d/2]

    # Create a vtkPoints container and store the points in it
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(p0)
    pts.InsertNextPoint(p1)
    pts.InsertNextPoint(p2)
    pts.InsertNextPoint(p3)
    pts.InsertNextPoint(p4)
    pts.InsertNextPoint(p5)
    pts.InsertNextPoint(p6)
    pts.InsertNextPoint(p7)

    # Add the points to the polydata container
    linesPolyData.SetPoints(pts)

    # Create 12 lines
    line0, line1, line2, line3,\
    line4, line5, line6, line7,\
    line8, line9, line10, line11 = [vtk.vtkLine() for i in range(12)]

    line0.GetPointIds().SetId(0, 0)  # the second 0 is the index of P0 in linesPolyData's points
    line0.GetPointIds().SetId(1, 1)  # the second 1 is the index of P1 in linesPolyData's points

    line1.GetPointIds().SetId(0, 2)  # the second 2 is the index of P2 in linesPolyData's points
    line1.GetPointIds().SetId(1, 3)  # the second 3 is the index of P3 in linesPolyData's points

    line2.GetPointIds().SetId(0, 4)  # the second 4 is the index of P4 in linesPolyData's points
    line2.GetPointIds().SetId(1, 5)  # the second 5 is the index of P5 in linesPolyData's points

    line3.GetPointIds().SetId(0, 6)  # the second 6 is the index of P6 in linesPolyData's points
    line3.GetPointIds().SetId(1, 7)  # the second 7 is the index of P7 in linesPolyData's points

    line4.GetPointIds().SetId(0, 0)  # the second 0 is the index of P0 in linesPolyData's points
    line4.GetPointIds().SetId(1, 2)  # the second 2 is the index of P2 in linesPolyData's points

    line5.GetPointIds().SetId(0, 1)  # the second 1 is the index of P1 in linesPolyData's points
    line5.GetPointIds().SetId(1, 3)  # the second 3 is the index of P3 in linesPolyData's points

    line6.GetPointIds().SetId(0, 4)  # the second 4 is the index of P4 in linesPolyData's points
    line6.GetPointIds().SetId(1, 6)  # the second 6 is the index of P6 in linesPolyData's points

    line7.GetPointIds().SetId(0, 5)  # the second 5 is the index of P5 in linesPolyData's points
    line7.GetPointIds().SetId(1, 7)  # the second 7 is the index of P7 in linesPolyData's points

    line8.GetPointIds().SetId(0, 0)  # the second 0 is the index of P0 in linesPolyData's points
    line8.GetPointIds().SetId(1, 4)  # the second 4 is the index of P4 in linesPolyData's points

    line9.GetPointIds().SetId(0, 1)  # the second 1 is the index of P1 in linesPolyData's points
    line9.GetPointIds().SetId(1, 5)  # the second 5 is the index of P5 in linesPolyData's points

    line10.GetPointIds().SetId(0, 2)  # the second 2 is the index of P2 in linesPolyData's points
    line10.GetPointIds().SetId(1, 6)  # the second 6 is the index of P6 in linesPolyData's points

    line11.GetPointIds().SetId(0, 3)  # the second 3 is the index of P3 in linesPolyData's points
    line11.GetPointIds().SetId(1, 7)  # the second 7 is the index of P7 in linesPolyData's points

    # Create a vtkCellArray container and store the lines in it
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line0)
    lines.InsertNextCell(line1)
    lines.InsertNextCell(line2)
    lines.InsertNextCell(line3)
    lines.InsertNextCell(line4)
    lines.InsertNextCell(line5)
    lines.InsertNextCell(line6)
    lines.InsertNextCell(line7)
    lines.InsertNextCell(line8)
    lines.InsertNextCell(line9)
    lines.InsertNextCell(line10)
    lines.InsertNextCell(line11)

    # Add the lines to the polydata container
    linesPolyData.SetLines(lines)

    #cube = vtk.vtkCubeSource()
    #cube.SetCenter(x, y, z)
    #cube.SetXLength(d)
    #cube.SetYLength(d)
    #cube.SetZLength(d)
    #cube.Update()

    #featureEdges = vtk.vtkFeatureEdges()
    #featureEdges.SetInputConnection(cube.GetOutputPort())
    #featureEdges.BoundaryEdgesOff()
    #featureEdges.FeatureEdgesOff()
    #featureEdges.ManifoldEdgesOff()
    #featureEdges.NonManifoldEdgesOff()
    #featureEdges.ColoringOn()

    #featureEdges.Update()

    cm = vtk.vtkPolyDataMapper()
    cm.SetInputData(linesPolyData)

    #cm.SetInputConnection(featureEdges.GetOutputPort())
    #cm.SetScalarModeToUseCellData()

    ca = vtk.vtkActor()
    ca.PickableOff() 
    ca.SetMapper(cm)
    ca.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d(color))
    #ca.GetProperty().SetColor((1, 0, 0))
    #ca.GetProperty().EdgeVisibilityOn()
    #ca.GetProperty().SetEdgeColor(vtk.vtkNamedColors().GetColor3d("Aqua"))

    ca.GetProperty().SetLineWidth(1.5)

    return ca
#------------------------------------------------------------------------
def draw_Text(ren, text, bbox):
    # follower, put text on graph
    z, y, x, d = bbox[:4]
    VectorText = vtk.vtkVectorText()
    VectorText.SetText(str(text) )   

    text_mapper = vtk.vtkPolyDataMapper() 
    text_mapper.SetInputConnection(VectorText.GetOutputPort())    

    follower = vtk.vtkFollower()
    follower.SetMapper(text_mapper)
    follower.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("MidnightBlue"))
    follower.SetPosition(x-d-4, y, z-d)
    follower.SetScale(8, 8, 8)
    follower.SetCamera(ren.GetActiveCamera())
    follower.PickableOff()  
    ren.AddActor(follower)

#---------------------------------------------------------------------------
def ren_lung(Renderer, lung_array, color):
    actor_lung = draw_lung(lung_array, color)
    # actor_lung.SetVisibility(True)
    Renderer.AddActor(actor_lung)
    return Renderer

def ren_nodule(Renderer, nodule_list, oribox):
    nodule_collection = vtk.vtkActorCollection()
    for i, nodule in enumerate(nodule_list):
        if i in range(2,5): # tmp
            continue
        nodule_collection.AddItem(draw_nodule(np.load(nodule), oribox))
    nodule_collection.InitTraversal()
    num = nodule_collection.GetNumberOfItems()

    print(num)
    for i in range(num):
        actor = nodule_collection.GetNextActor()
        Renderer.AddActor(actor)
        
    # nodule_collection.InitTraversal()
    # actor = nodule_collection.GetNextActor()
    # actor.SetVisibility(False)
    return Renderer

def ren_bbox(Renderer, bbox_array):
    for i, bbox in enumerate(bbox_array):
        if i in range(2,5): # tmp
            continue
        Renderer.AddActor(draw_bbox(bbox))
        if i in [1,5]:
            txt = "Malignant" 
        else :
            txt = "Benign"# if np.random.rand() > 0.5 else "Malignant" 
        draw_Text(Renderer, txt, bbox)
    return Renderer
#---------------------------------------------------------------------------
def lung_nodule_bbox(Renderer, lung_array, color, show_lung=True, show_nodule=False, show_bbox=False):
    if show_lung == True:
        # Renderer = ren_lung(Renderer, lung_array)
        ren_lung(Renderer, lung_array, color)
    #if show_nodule == True:
        # Renderer = ren_nodule(Renderer, nodule_list, oribox)
        # ren_nodule(Renderer, nodule_list, oribox)
    #if show_bbox == True:
        # Renderer = ren_bbox(Renderer, bbox_array)
        # ren_bbox(Renderer, bbox_array)



    # return Renderer


def lung_nodule_bbox2(Renderer, lung_array, oribox, bbox_array, show_lung=True, show_nodule=True,
                     show_bbox=False):
    if show_lung == True:
        # Renderer = ren_lung(Renderer, lung_array)
        ren_lung(Renderer, lung_array)

    if show_bbox == True:
        # Renderer = ren_bbox(Renderer, bbox_array)
        ren_bbox(Renderer, bbox_array)


#----------------------------------------------------------------------------
# Reset Camera view point
def reset(Camera):
    Camera.SetFocalPoint(0, 0,  0)
    Camera.SetPosition(0, -1, 0)
    Camera.SetViewUp(0, 0, -1)
#----------------------------------------------------------------------------
# show x-y-z Axes grids
def GetCubeAxes(lung_array):
    w, h, d = lung_array.shape[:3]
    cubeAxes = vtk.vtkCubeAxesActor()
    cubeAxes.SetBounds(0, d, 0, h, 0, w)
    
    cubeAxes.SetZTitle('Z')
    cubeAxes.SetYTitle('Y')
    cubeAxes.SetXTitle('X')
    
    cubeAxes.DrawXGridlinesOn()
    cubeAxes.DrawYGridlinesOn()
    cubeAxes.DrawZGridlinesOn()
    
    cubeAxes.GetTitleTextProperty(0).SetColor(0,0,0)
    cubeAxes.GetLabelTextProperty(0).SetColor(0,0,0)
    
    cubeAxes.GetTitleTextProperty(1).SetColor(0,0,0)
    cubeAxes.GetLabelTextProperty(1).SetColor(0,0,0)
    
    cubeAxes.GetTitleTextProperty(2).SetColor(0,0,0)
    cubeAxes.GetLabelTextProperty(2).SetColor(0,0,0)
    
    cubeAxes.GetXAxesLinesProperty().SetColor(0,0,0)
    cubeAxes.GetYAxesLinesProperty().SetColor(0,0,0)
    cubeAxes.GetZAxesLinesProperty().SetColor(0,0,0)
    
    cubeAxes.GetXAxesGridlinesProperty().SetColor(0,0,0)
    cubeAxes.GetYAxesGridlinesProperty().SetColor(0,0,0)
    cubeAxes.GetZAxesGridlinesProperty().SetColor(0,0,0)
    
    cubeAxes.SetGridLineLocation(cubeAxes.VTK_GRID_LINES_FURTHEST)
    cubeAxes.XAxisMinorTickVisibilityOff()
    cubeAxes.YAxisMinorTickVisibilityOff()
    cubeAxes.ZAxisMinorTickVisibilityOff()

    return cubeAxes

#------------------------------------------------------------------
# mouse event (get actor info.)
class MouseInteractorActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)

        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()

    def leftButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()
        # picker = vtk.vtkPropPicker()
        picker = vtk.vtkPointPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        actor = picker.GetActor()
        # print(actor)
        # SetVisibility False
        # actor.SetVisibility(False)

        self.OnLeftButtonDown()

        return 

#------------------------------------
# # 切面
# def GetPlaneWidget(Origin,pt1,pt2,Data,Color):
#     wid = vtk.vtkImagePlaneWidget()
#     wid.SetInputData(Data.GetOutput())
    
#     wid.TextureVisibilityOn()
    
#     #邊框的Property
#     wid.GetPlaneProperty().SetColor(Color)
#     wid.GetPlaneProperty().SetLineWidth(5)
    
#     #隱藏切面圖片(才能只看到邊框)
#     wid.GetTexturePlaneProperty().SetOpacity(0)
    
#     #讓Plane可以跑出Volume的範圍(不會自動校正位置)
#     wid.RestrictPlaneToVolumeOff()
    
#     #切面圖片的插值法
#     wid.SetResliceInterpolateToLinear()
    
#     #切面的位置(Origin指向pt1及pt2形成兩個邊，兩個邊可以圍出一個平行四邊形)
#     wid.SetOrigin(Origin)
#     wid.SetPoint1(pt1)
#     wid.SetPoint2(pt2)
    
#     #更新Plane widget
#     wid.UpdatePlacement()
    
#     return wid

#----------------------------------------
# Remove_Actor = ren.GetActors()
# for actor in Remove_Actor:
#     ren.RemoveActor(actor)
# ren.RemoveAllViewProps()

#import cc3d
#def get_clean_surface(clean):
#    labels = cc3d.connected_components(clean)
#    clean_surface = np.ones_like(clean, dtype=np.uint8)
#    clean_surface = clean_surface - labels
#    clean_surface *= 255
#    return clean_surface