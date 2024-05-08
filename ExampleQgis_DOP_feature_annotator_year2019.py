# %%
import os
# %%
from qgis.utils import iface
from qgis.PyQt.QtWidgets import QAction,QLabel,QLineEdit, QMessageBox, QPushButton
from qgis.PyQt.QtCore import QSize, QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsVectorLayer,QgsFeature, QgsProject, NULL, QgsVectorFileWriter
from qgis.core import QgsField, edit, QgsRasterLayer,QgsGeometry,QgsDefaultValue
project_instance = QgsProject.instance()
root = project_instance.layerTreeRoot()

def is_Layer(layer_name):
    return len(project_instance.mapLayersByName(layer_name)) != 0
#results_path = "E:\\Uni\\Research\\row_detector\\_data\\validation\\5b8701a4-b7b4-11eb-97ea-0242ac110003"

#id = "5379d016-7ead-11ec-aab7-ff7719ff8ac2"
results_path = os.path.join("E:","Uni","Research",
    "row_detector","_data","validation","invekos2019_img2019",
    "3af40c6a-8915-11ec-aab7-ff7719ff8ac2"
    #"5379d016-7ead-11ec-aab7-ff7719ff8ac2_done"
    
    )
    

#id = "235085be-8511-11ec-aab7-ff7719ff8ac2"
#results_path = os.path.join("E:","Uni","Research",
#    "row_detector","_data","testsets","invekos2019_img2019",id)


#path_to_field = os.path.join(results_path,"field_shape_valid.shp")
path_to_field = os.path.join(results_path,"sample_fields.shp")

#path_to_measurements = os.path.join(results_path,"measurments.shp")

print(path_to_field)


def load_or_connect_layer(name,path):

    if not is_Layer(name):
        layer = QgsVectorLayer(path, name, "ogr")
        if not layer.isValid():
            print(name," layer failed to load!")
        else:
            project_instance.addMapLayer(layer)
            return layer
    else:
        return project_instance.mapLayersByName(name)[0]

vlayer = load_or_connect_layer('Fields', path_to_field)
#measurments_layer = load_or_connect_layer('Measurments',path_to_measurements)        
print(vlayer)
# Zoom to layer    
vlayer.selectAll()
canvas = iface.mapCanvas()
canvas.zoomToSelected()
# Set color to transparent
canvas.setSelectionColor( QColor("Transparent") )
# get the layerTreeCanvas Bridge, its a way to interact with the 
# layers, that is automaticly updated
bridge = iface.layerTreeCanvasBridge().rootGroup()

# enable the Layer 
bridge.setHasCustomLayerOrder (True)

#dop_years = range(1996,2020)
dop_years = range(2019,2020)
count = 0
desired_amount_triangles_per_field = 3
shape_id_string = 'ID'

results_layer_prefixes = ['new_field_boundaries','triangle_measurments']
annotation_field_base_names = ["_statu","_shdif","_nvis"]


ID = 0
YEAR = 2019

def zoom_next_shape():
    global ID
    ID += 1
    zoom_shape(ID)
    
def zoom_prev_shape():
    global ID
    ID -= 1
    zoom_shape(ID)
    
def zoom_shape(ID,reset_year_tick = True):
    print("zooming to ",ID)
    vlayer.removeSelection()
    
    #canvas.zoomToFeatureIds(vlayer, [ID])
    vlayer.selectByIds([ID])
    canvas.zoomToSelected(vlayer)
    # Set color to transparent
    canvas.setSelectionColor( QColor("Transparent") )
    shape_text_id.setText(str(ID))
    if reset_year_tick:
        reset_year()
    
def zoom_next_year(restart_anotation = False):
    global YEAR
    YEAR += 1
    insert_layer_in_order()
    if restart_anotation:
        annotate_selected()

def zoom_next_field_anotate(restart_anotation = False):
    global ID
    ID += 1
    zoom_shape(ID)
    insert_layer_in_order()
    if restart_anotation:
        annotate_selected()
   
def zoom_prev_year():
    global YEAR
    YEAR -= 1
    insert_layer_in_order()

def zoom_year(year):
    global YEAR
    YEAR = year
    insert_layer_in_order()
    
def reset_year():
    global YEAR
    YEAR = dop_years[0]
    year_text_id.setText(str(YEAR))
    insert_layer_in_order(rezoom=False)
    
def reset_id():
    global ID
    ID = 0
    shape_text_id.setText('0')
    #zoom_shape(ID)
    canvas.zoomToFeatureIds(vlayer, [ID])
    print('initially zoomed')
    
def insert_layer_in_order(other_layer = False,background=True, rezoom=True):
    
   
    # gather the order of displayed Layers from the bridge
    order = bridge.customLayerOrder()
    
    
    
    if background:
        Background_vector_layer = project_instance.mapLayersByName( 'Background' )[0]
        Background_vector_layer = root.findLayer(Background_vector_layer.id())
        # insert the Background Layer in the first position
        order.insert( 0, order.pop( order.index(Background_vector_layer.layer())) )
    
    # insert the DOP 
    # gather current DOP with current globally set YEAR
    year_text_id.setText(str(YEAR))
    name = f'Historische DOP {YEAR}'
    layerDOP = project_instance.mapLayersByName( name )[0]
    # access the layer in the treeroot
    layerDOP = root.findLayer(layerDOP.id())
    # ensure Layer Visibility, should be alway be True
    layerDOP.setItemVisibilityChecked(True)
    order.insert( 0, order.pop( order.index(layerDOP.layer())) ) # vlayer to the top
    #print("inserting")
    
    if other_layer:
        #print("Trigger Warning")
        # access the layer in the treeroot
        myLayer = root.findLayer(other_layer.id())
    
        # ensure Layer Visibility, should be alway be True
        myLayer.setItemVisibilityChecked(True)
        # insert the desired DOP Layer in the first position
        order.insert( 0, order.pop( order.index(myLayer.layer())) ) # vlayer to the top
    
    # insert the Field Layer in the first position
    field_layer = root.findLayer(vlayer.id())
    order.insert( 0, order.pop( order.index(field_layer.layer())) ) # vlayer to the top
    # finally set the order
    print(order[0])
    bridge.setCustomLayerOrder( order )
    canvas.refresh()
    #print("canvas.isFrozen()",canvas.isFrozen())
    if canvas.isFrozen():
        canvas.freeze(False)
    #print("canvas.isFrozen()",canvas.isFrozen())
    #print("canvas.isParallelRenderingEnabled(self) ",canvas.isParallelRenderingEnabled() )
    #print("canvas.layer(index: 0)",canvas.layer(index= 0))
    #canvas.mapCanvasRefreshed.connect(lambda:print("refreshed"))
    #print("canvas.renderStarting",canvas.renderStarting)
    #QApplication.processEvents()
    #this results in a Blank Canvas behind the loaded DOP
    # If the Visability of the current Layer is to be ensured,
    # its best Praktice to change the Layer Order
    # this prevents rerendering in QGIS in allows for quicker feature annotation
    
    # a rezooming is nececary, since layer order changing of the field layer,
    # deselects the previously selected
    if rezoom:
        zoom_shape(ID,reset_year_tick = False)




def continue_ensurence(to_draw_on,next_action,return_to_previous_action,args):
    to_draw_on.editingStopped.disconnect()
    to_draw_on.committedFeaturesAdded.disconnect()
    mb = QMessageBox()
    mb.setText('Are you shure to continue?')
    mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
    return_value = mb.exec()
    
    if return_value == QMessageBox.Yes:
        next_action()
    elif return_value == QMessageBox.No:
        print("return_to_previous_action(*args)",args)
        return_to_previous_action(*args)

def draw_something(on_wich_layer_type,selection=False):
    global to_draw_on
    print(f'{on_wich_layer_type}_{YEAR}')
    to_draw_on = project_instance.mapLayersByName( f'{on_wich_layer_type}_{YEAR}' )[0]
    insert_layer_in_order(to_draw_on)
    iface.setActiveLayer(to_draw_on) 
    to_draw_on.startEditing()
    #TODO:Toogle editing
    if on_wich_layer_type=='triangle_measurments':
        # Signal will be executed without caring about if feature added has been committed
        to_draw_on.featureAdded.connect(lambda:check_for_enough_features(to_draw_on))
        # connect a signal to listen to feature deletion. this ensures the clobal count is correct
        to_draw_on.featureDeleted.connect(reduce_feature_count)
       
    if on_wich_layer_type=='new_field_boundaries':
        # this seems to stay connected, resulting in errors
        to_draw_on.editingStopped.connect(lambda: continue_ensurence(to_draw_on,line_questions,draw_something,[on_wich_layer_type,selection]))
        to_draw_on.committedFeaturesAdded.connect(lambda:line_questions(to_draw_on = to_draw_on))
        #TODO connection has to be severed
        # is possible by following command, yet need to be called from inside the triggered function
        # isSignalConnected receivers()? senderSignalIndex
        
    
def check_if_annotated_per_shape(selection):
    for year in dop_years:
        print("year",year)
        print("selection[f]",selection[f"{year}_statu"])
        if selection[f"{year}_statu"]== NULL:
            print("Trigger Warning")
            return year  
            
    
    return False     

def check_if_annotated():
    broken = False
    #reset_id()
    for id in range(vlayer.featureCount()):
        print("id",id)
        if not broken:
            vlayer.selectByIds([id])
            selection = vlayer.selectedFeatures()[0]
            possibly_a_year = check_if_annotated_per_shape(selection)
            print("possibly_a_year",possibly_a_year)
            if possibly_a_year:
                broken = True
                print("zooming !")
                zoom_year(possibly_a_year)
                global ID
                ID = id
                zoom_shape(ID)
                
        
            
            
        
def annotate_per_year(selection):
    
    print("selection in annotate year",selection)
    
    # check if annotation has not been cperformed
    # and additional redo annotation
    mb = QMessageBox()
    mb.setText('Is DOP available')
    
    mb.setStandardButtons(QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)
    return_value = mb.exec()
    
    if return_value == QMessageBox.Yes:
        
        """
        mbShape_equality = QMessageBox()
        mbShape_equality.setText('Are the Fieldboundaries different?')
        mbShape_equality.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
        return_value_shape_equality = mbShape_equality.exec()
        
        if return_value_shape_equality == QMessageBox.Yes:
            selection[f"{YEAR}_shdif"] = "1"
            with edit(vlayer):
                vlayer.updateFeature(selection)
            #vlayer.commitChanges()
            # ask wether its visible
            mbShape_equality = QMessageBox()
            mbShape_equality.setText('Are new Fieldboundaries visable?')
            mbShape_equality.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
            return_value_shape_equality = mbShape_equality.exec()
            if return_value_shape_equality == QMessageBox.Yes:
                selection[f"{YEAR}_nvis"] = "1"
                with edit(vlayer):
                    vlayer.updateFeature(selection)
                mbField = QMessageBox()
                mbField.setText('Please draw the new boundaries')
                mbField.exec()
                draw_something('new_field_boundaries',selection)
            elif return_value_shape_equality == QMessageBox.No:
                selection[f"{YEAR}_nvis"] = "0"
                with edit(vlayer):
                    vlayer.updateFeature(selection)
                #vlayer.commitChanges()
                #TODO better would be a default setting for the other parameters
                line_questions(selection)
            
        elif return_value_shape_equality == QMessageBox.No:
            selection[f"{YEAR}_shdif"] = "0"
            with edit(vlayer):
                vlayer.updateFeature(selection)
            #vlayer.commitChanges()
            line_questions(selection)
        """    

        line_questions(selection)
        
        
    elif return_value == QMessageBox.No:
        print('You pressed Cancel')
        #vlayer.changeAttributeValue(fid= ID, field: int, newValue: Any
        selection[f"{YEAR}_statu"] = "0"
        with edit(vlayer):
            vlayer.updateFeature(selection)
        #vlayer.commitChanges()
        if YEAR<dop_years[-1]:
            zoom_next_year(restart_anotation = True)
    elif return_value == QMessageBox.Cancel:
        print('You canceled the Loop')
        pass
        
def line_questions(selection=False,to_draw_on = False):
    if to_draw_on:
        to_draw_on.editingStopped.disconnect()
        to_draw_on.committedFeaturesAdded.disconnect()
    print("selection in line question",selection)
    insert_layer_in_order()
    #if not selection:
    vlayer.selectByIds([ID])
    selection = vlayer.selectedFeatures()[0]
    mbLines = QMessageBox()
    mbLines.setText('Are there regular lines?')
    mbLines.setStandardButtons(QMessageBox.Yes | QMessageBox.No |QMessageBox.Cancel )
    return_value_lines = mbLines.exec()
    
    if return_value_lines == QMessageBox.Yes:
        
        selection[f"{YEAR}_statu"] = "3"
        with edit(vlayer):
            vlayer.updateFeature(selection)
        #vlayer.commitChanges()
        mbLines = QMessageBox()
        #mbLines.setText('Pls Draw some Triangles')
        #mbLines.exec()
        
        draw_something('triangle_measurments')
        
    elif return_value_lines == QMessageBox.No:
        selection[f"{YEAR}_statu"] = "2"
        with edit(vlayer):
            vlayer.updateFeature(selection)
        #vlayer.commitChanges()
        #if YEAR<dop_years[-1]:
        #    zoom_next_year(restart_anotation = True)
        zoom_next_field_anotate(restart_anotation = True)
    
    #elif return_value_lines == QMessageBox.Cancel:
    #    print('You canceled the Loop')
    #    pass

        
def annotate_selected():
    vlayer.selectByIds([ID])
    selection = vlayer.selectedFeatures()[0]
    line_questions(selection)
    #annotate_per_year(selection)
    
    
def check_layer_existance_or_create_andor_load(name):
    dop_name = f'Historische DOP {2019}'
    DOP_example = project_instance.mapLayersByName( dop_name )[0]
    import os
    path = os.path.join(results_path,name+'.shp')
    if not is_Layer(name):
        if not os.path.isfile(path):
            layer = QgsVectorLayer("Polygon", "name", "memory",crs=DOP_example.crs())
            coordinateTransformContext=QgsProject.instance().transformContext()
            print("coordinateTransformContext",coordinateTransformContext)
            #pr = layer.dataProvider()
            #options = QgsVectorFileWriter.SaveVectorOptions()
            #options.driverName = "ESRI Shapefile"       
            #options.skipAttributeCreation = True
            #QgsVectorFileWriter.writeAsVectorFormatV2(layer,path,coordinateTransformContext)
            QgsVectorFileWriter.writeAsVectorFormat(layer,path,'utf-8',DOP_example.crs(),driverName='ESRI Shapefile',skipAttributeCreation = True)
            #print("layer.transformContext()",layer.transformContext())
            #QgsVectorFileWriter.writeAsVectorFormatV2(layer,path,'utf-8',DOP_example.crs(),driverName='ESRI Shapefile',skipAttributeCreation = True)
            
        else:
            load_or_connect_layer(name,path)
    
            
    
    
def initialise_results_layers():
    for year in dop_years:
        for prefix in results_layer_prefixes:
            check_layer_existance_or_create_andor_load(prefix+f'_{year}')
    
    
    
def add_year_and_shape_status_to_layer():
    with edit(vlayer):
        for year in dop_years:
            # Add attribute
            for base_name in annotation_field_base_names: 
                vlayer.addAttribute(QgsField(str(year)+base_name, QVariant.String))
    for year in dop_years:  
        for base_name in annotation_field_base_names:   
            
            idx = vlayer.fields().indexFromName(str(year)+base_name)
            vlayer.setDefaultValueDefinition(idx, QgsDefaultValue('-1'))
        
def add_wms_layers():
    for year in dop_years:
        layer_name = f'Historische DOP {year}'
        print("is_Layer(layer_name)",is_Layer(layer_name))
        if not is_Layer(layer_name):
            urlWithParams = 'contextualWMSLegend=0&crs=EPSG:25832&dpiMode=7&featureCount=10&format=image/png&layers=nw_hist_dop_%s&styles=default&url=https://www.wms.nrw.de/geobasis/wms_nw_hist_dop'%(year)
            layerDOP = QgsRasterLayer(urlWithParams,layer_name , 'wms')
            #rlayer.isValid()
            project_instance.addMapLayer(layerDOP)
            myLayer = root.findLayer(layerDOP.id())
            myLayer.setItemVisibilityChecked(True)
    if not is_Layer("Background"):
        Background_vector_layer = QgsVectorLayer("Polygon", "Background", "memory",crs=layerDOP.crs())
        pr = Background_vector_layer.dataProvider()
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromWkt(layerDOP.extent().asWktPolygon()))
        pr.addFeature(f)
        Background_vector_layer.updateExtents() 
        project_instance.addMapLayer(Background_vector_layer)
    reset_year()
        
        
def check_for_enough_features(to_draw_on):
    #TODO count does not react on feature removal
    # a spatial join would be to cost intensive
    # ids per added feauture are to human labor intensive
    # implement count connect to feature removal
    global count
    count = count+1
    if count==desired_amount_triangles_per_field:
        
        # disconnect the signals to prevent signaling on next field-feature in same year
        to_draw_on.featureAdded.disconnect()
        to_draw_on.featureDeleted.disconnect()
        count = 0
        #if YEAR<dop_years[-1]:
        #    zoom_next_year(restart_anotation = True)
        zoom_next_field_anotate(restart_anotation = True)
        
    
def reduce_feature_count():
    global count
    count = count-1
       

## ACTIONS
zoom_next_shape_action = QAction("Next Field")
zoom_next_shape_action.triggered.connect(zoom_next_shape)

zoom_prev_shape_action = QAction("Previous Field")
zoom_prev_shape_action.triggered.connect(zoom_prev_shape)

zoom_next_year_action = QAction("Next Year")
zoom_next_year_action.triggered.connect(zoom_next_year)

zoom_prev_year_action = QAction("Previous Year")
zoom_prev_year_action.triggered.connect(zoom_prev_year)

initiate_attributes_action = QAction("Initiate Attributes")
initiate_attributes_action.triggered.connect(add_year_and_shape_status_to_layer)

add_wms_action = QAction("Add historical NRW DOP")
add_wms_action.triggered.connect(add_wms_layers)

initialise_results_layers_action = QAction("Initialise Resultslayers")
initialise_results_layers_action.triggered.connect(initialise_results_layers)

annotate_action = QAction("Annotate Selected")
annotate_action.triggered.connect(annotate_selected)

check_annotation_action = QAction("Search for next feature and year to be annotated")
check_annotation_action.triggered.connect(check_if_annotated)


shape_label = QLabel("ID:")
shape_text_id = QLineEdit()
shape_text_id.setMaximumSize(QSize(40, 100))
shape_text_id.setText('0')

year_label = QLabel("Year:")
year_text_id = QLineEdit()
year_text_id.setMaximumSize(QSize(40, 100))
year_text_id.setText(str(dop_years[0]))

## TOOLBAR
#anotate_toolbar = iface.addToolBar("Anotate Features")
#anotate_toolbar.addAction(initialise_results_layers_action)
#anotate_toolbar.addAction(initiate_attributes_action)
#anotate_toolbar.addAction(add_wms_action)
#anotate_toolbar.addAction(annotate_action)
#anotate_toolbar.addAction(zoom_prev_year_action)
#anotate_toolbar.addAction(zoom_next_year_action)

anotate_toolbar = iface.addToolBar("Anotate Features")
anotate_toolbar.addWidget(shape_label)
anotate_toolbar.addWidget(shape_text_id)
anotate_toolbar.addAction(zoom_prev_shape_action)
anotate_toolbar.addAction(zoom_next_shape_action)
anotate_toolbar.addAction(initiate_attributes_action)
anotate_toolbar.addAction(add_wms_action)
anotate_toolbar.addAction(initialise_results_layers_action)
anotate_toolbar.addAction(annotate_action)
anotate_toolbar.addWidget(year_label)
anotate_toolbar.addWidget(year_text_id)
anotate_toolbar.addAction(zoom_prev_year_action)
anotate_toolbar.addAction(zoom_next_year_action)
anotate_toolbar.addAction(check_annotation_action)


reset_id()
#vlayer.attributeValueChanged.connect(lambda:print("attributeValueChanged"))
#vlayer.committedFeaturesAdded.connect(lambda:print("committedFeaturesAdded"))
#vlayer.updatedFields.connect(lambda:print("updatedFields"))
#vlayer.selectionChanged.connect(lambda:print("selectionChanged"))


#iface.layerTreeView().currentLayerChanged.connect(reset_id)
# perhaps check on measurments layer, to reakt on change, for switch to next year

#beforeModifiedCheck
#editCommandEnded