# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:24:03 2023

@author: gaurr
"""

## Importing necessary packages

import numpy as np
import glob
from PIL import Image, ImageDraw, ImageGrab
import io
import csv
import pandas as pd
import os.path
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import geopandas as gpd

import Feature_Labeling_Functions as func
import Feature_Labeling_Variables as var

import sys
sys.path.append(r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Reconstruction\PhotogrammetryAnalysis-master')
# import SK_ring-relabelling-secondattempt as recon



# %% Defining window objects and layout

# sg.theme_previewer() ## Use to see all the colour themes available
sg.theme('LightPurple') ## setting window colour scheme
sg.set_options(dpi_awareness=True)

## Top menubar options
menu_butts = [['File', ['New', 'Open', 'Save', 'Exit', ]], ['Edit', ['Copy', '&Undo point', 'Resize Image', 'Change Canvas Size'], ],  ['Help', 'About...'], ]
menubar = [[sg.Menu(menu_butts)],]

## Window column 1: list of files in chosen folder

file_list_col = [
    [
     sg.Text("Image folder"),
     sg.In(size=(15,1), enable_events =True, key="-FOLDER-"),
     sg.FolderBrowse(),
     ],
    [sg.Listbox(values=[], enable_events=True, size=(25,20), key="-FILE LIST-")
     ], ## displays a list of paths to the images you can choose from to display
    ]

mov_col = [[sg.T('Choose what you want to do:', enable_events=True)],
           [sg.R('Draw PMT points', 1,  key='-PMT_POINT-', enable_events=True)],
           [sg.R('Draw bolt points', 1,  key='-BOLT_POINT-', enable_events=True)],
           [sg.R('Erase item', 1, key='-ERASE-', enable_events=True)],
           [sg.R('Modify label', 1, key='-MODIFY-', enable_events=True)],
           [sg.R('Erase all', 1, key='-CLEAR-', enable_events=True)],
           [sg.R('Send to back', 1, key='-BACK-', enable_events=True)],
           [sg.R('Bring to front', 1, key='-FRONT-', enable_events=True)],
           [sg.R('Move Everything', 1, key='-MOVEALL-', enable_events=True)],
           [sg.R('Move Stuff', 1, key='-MOVE-', enable_events=True)],
           [sg.R("Auto-label", 1, key='-LABELING-', enable_events=True)],
           [sg.R('Cursor Position (Click & Hold)', 1,  key='-SCAN_POSITION-', enable_events=True)],
           ]

column = [[sg.Graph(canvas_size = (var.width, var.height), graph_bottom_left = (0,0), graph_top_right = (var.width,var.height), 
key = "-GRAPH-", ## can include "change_submits = True"
background_color = 'white', expand_x =True, expand_y = True, enable_events = True, drag_submits = True, right_click_menu=[[],['Erase Item',]])]]

image_viewer_col_2 = [
    [sg.Text("Choose an image from list on the left: ")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Column(column, size=(var.column_width, var.column_height), scrollable = True, key = "-COL-")],
    [sg.Text(key="-INFO-", size=(90,1))],
    ]

post_process_col= [
    [sg.Column(mov_col)],
    [sg.Button("Save Annotations", size = (15,1), key="-SAVE-")],
    [sg.Button("Undistort", size = (15,1), key="-UNDISTORT-")],
    [sg.Button("Plot Labels", size = (15,1), key="-PLOT_LABEL-"), sg.Button("Remove Labels", size = (18,1), key="-ERASE_LABEL-")],
    [sg.Button("Write CSV", size = (15,1), key="-CSV-")],
    [sg.Button("Reconstruct", size=(15,1), key="-RECON-")],
    [sg.Button('Autolabel', size =(15,1), key='-AUTO_LABEL-')],
    [sg.Button('Zoom In'), sg.Button('Zoom Out')],
    [sg.Button("Shift R"), sg.Button("Shift L")],
    [
     sg.Text("Choose a file of overlay points: "),
     sg.In(size=(18,1), enable_events =True, key="-OVERLAY-"),
     sg.FileBrowse()],
     ]
    


## Main function
def main():
    
    # ------ Full Window Layout
    layout= [
        [menubar, sg.Column(file_list_col), sg.VSeperator(), sg.Column(image_viewer_col_2), sg.VSeperator(), sg.Column(post_process_col),
         [sg.Button("Prev", size=(10,1), key="-PREV-"), sg.Button("Next", size=(10,1), key="-NEXT-")],]] 
    
    window = sg.Window("Image Labeling GUI", layout, resizable=True) ## putting together the user interface
    
    
    location = 0
    
    graph = window["-GRAPH-"] 

    name = None
    while name is None:    
        name = sg.popup_get_text('Please enter your initials', title="Name initials")
        
    Img_ID = None
    
    dragging = False
    made_point = False
    start_pt = end_pt = None
    ids = None ## used to specify which figure on sg.Graph to delete
    first_pmt = None
    pts_dir = None
    df = None
    labels = []
    
    while True:
        event, values = window.read()
        ## 'event' is the key string of whichever element user interacts with
        ## 'values' contains Python dictionary that maps element key to a value
        
        if event == "Exit" or event == sg.WIN_CLOSED: ## end loop if user clicks "Exit" or closes window
            break;
        
        if event == "-UNDISTORT-":
            undistort = True
        else:
            undistort = False
        
        # Folder name filled in, we'll now make a list of files in the folder
        if event == "-FOLDER-": ## checking if the user has chosen a folder
            fnames = func.parse_folder(window, values)
                        
        elif event == "-FILE LIST-": ## User selected a file from the listbox
            
            if len(values["-FOLDER-"]) == 0 : ## If user selected listbox before browsing folders
                select_folder = sg.popup_ok("Please select a folder first.")
                        
            elif len(values["-FOLDER-"]) != 0:
                
                if ids is not None:
                    # graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                    graph.erase()
                    
                image, pil_image, filename, ids, newcameramtx = func.disp_image(window, values, fnames, location=0, undistort=undistort)
                first_pmt = None ## to allow autolabeling again
                window.refresh()
                window["-COL-"].contents_changed()
                
                try:
                    ## Tries to overlay points if the image has associated coordinate file
                    pts_dir = os.path.join(os.path.dirname(values["-FOLDER-"]), 'points')
                    pts_file = os.path.basename(filename).split('.')[0]
                    pts_fname = os.path.join(pts_dir, pts_file) + ".txt" 
    
                    df = func.autoload_pts(pts_fname, name, var.mtx, undistort)
                    print("Autoloaded the points.")
                    Img_ID = df['Img'].iloc[0]
                    func.draw_pts(graph, df, undistort)
                    print("Drawing the points")
                    func.erase_labels(graph, labels)             
                    labels = func.plot_labels(graph, df, undistort)
                    print("Plotting the labels.")
                except Exception as e:
                    print(e)
                    print("No associated points file found.")
   
        if event == "-NEXT-" or event == "-PREV-":
            
            if ids is not None:
                # graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                graph.erase()
            
            location=1
            if event == "-PREV-":
                location=-1

            image, pil_image, filename, ids, newcameramtx = func.disp_image(window, values, fnames, location=location, undistort=undistort)
            first_pmt = None ##to allow autolabeling again
            window.refresh()
            window["-COL-"].contents_changed()
            
            try:
                pts_file = os.path.basename(filename).split('.')[0]
                pts_fname = os.path.join(pts_dir, pts_file) + ".txt"             
                df = func.autoload_pts(pts_fname, name, newcameramtx, undistort)
                Img_ID = df['Img'].iloc[0]
                func.draw_pts(graph, df, undistort)
                func.erase_labels(graph, labels)                
                labels = func.plot_labels(graph, df, undistort)
            except Exception as e:
                print(e)
                print("No associated points file found.")
        
## =============== Annotation Features ========================            
            
        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='cross')
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr') 
        
        if event == "-GRAPH-":

            x, y = values["-GRAPH-"]
            window["-INFO-"].update(value=f'({x}, {y})')

            if not dragging:
                dragging = True
                figures = graph.get_figures_at_location((x,y))

                for fig in figures:
                    start_pt = func.get_marker_center(graph, fig)
            else:
                end_pt = x, y
                for fig in figures[1:]:
                    end_pt = func.get_marker_center(graph, fig)      

            if values['-MOVE-']:

                ## ignoring the background image from the tuple of objects that can be dragged
                if len(figures)>1: 
                    fig = figures[1]
                    curr_x, curr_y = func.get_marker_center(graph, fig)
                    
                    graph.move_figure(fig, x - curr_x, y - curr_y)  ## start with marker centered on cursor
                      
                    graph.update()
            
            elif (values["-PMT_POINT-"] or values["-BOLT_POINT-"]) and not made_point:
                
                try:
                    ## drawing PMT point
                    if values["-PMT_POINT-"]:                    
                        pmt_id = sg.popup_get_text('Please enter PMT ID', title="Adding PMT")
                        if pmt_id:
                            df = func.make_pmt(df, pmt_id, x, y, name)
                            graph.draw_point((x,y), color = 'red', size=8)

                    ## drawing bolt point
                    elif values["-BOLT_POINT-"]:

                        ## checks which pmt the bolt belongs to and returns ID of the PMT
                        ## along with the angle between the dynode and the bolt    
                            df = func.make_bolt(df, x, y, name)
                            graph.draw_point((x,y), color = 'yellow', size=6)
                    
                    window["-INFO-"].update(value=f'Made point {df["ID"].iloc[-1]} at ({x}, {y})')
                    made_point = True

                except Exception as e:
                    print(e)
                    window["-INFO-"].update(value=f'Failed point at ({x}, {y})')

    ## ================= AUTO LABELING ===============================
            ## User clicks on a PMT and program gets the PMTs coordinates from the graph.
            elif values["-LABELING-"]:

                if first_pmt == None:
                    #sg.popup_ok(("You selected your first PMT."))

                    try:
                        # get figure at location
                        fig = graph.get_figures_at_location(values['-GRAPH-'])[1]
                        curr_x, curr_y = func.get_marker_center(graph, fig)
                        print("Coordinates of selected PMT: ", curr_x, curr_y)

                        # look for curr_x and curr_y in the dataframe
                        first_pmt = func.get_pmt(df, curr_x, curr_y)

                        # insert column after df['ID'] column with name 'Labels' in the dataframe
                        df.insert(2, 'Labels', 'None')

                        # get the index of the first pmt and get user input for this PMT's label. Put input into df['Labels'] column at index
                        index = df[df['ID'] == first_pmt].index[0]
                        print("Index of selected PMT in dataframe: ", index)
                        label = int(sg.popup_get_text('Please enter label', title="Adding Label"))
                        df.at[index, 'Labels'] = label

                        # finds PMT in dataframe and the other PMTs in the same row and column. Labels the row and column PMTs.
                        df, new_ref, row, column = func.autolabel(df, first_pmt, label)

                        print("Row and column ", row, column)
                        col_len = len(column)

                        count = 0
                        while count < col_len:

                            # look for the next PMT in the 'column'
                            index = df[df['ID'] == column[0]].index[0]

                            # get the ID of the PMT at the index and its label
                            new_ref = df['ID'].iloc[index]
                            print("New reference PMT: ", new_ref)
                            new_label = df['Labels'].iloc[index]
                            print("New reference PMT label: ", new_label)

                            df, new_ref, row, column = func.autolabel(df, new_ref, new_label)
                            print("Row and column ", row, column)
                            count +=1
                            print("Count: ", count)


                        print("Final dataframe with all labels. ", df.to_string())

                        # remove '-LABELING-' button selection
                        window["-LABELING-"].update(value=False)

                        ## get the number of PMTs with labels that are not None
                        num_labeled = len(df[df['Labels'] != 'None'])
                        print("Number of labeled PMTs: ", num_labeled)

                        # create new plot using matplotlib with only PMTs using their coordinates from the dataframe and plot the associated labels as text
                        # get indicies of the PMTs, 'ID' column will end with '00'
                        pmt_indicies = df[df['ID'].str.endswith('00')].index

                        # get the x and y coordinates of the PMTs
                        x = np.array(df['X'].iloc[pmt_indicies])
                        y = np.array(df['Y'].iloc[pmt_indicies])

                        # get the labels of the PMTs
                        pmt_labels = np.array(df['Labels'].iloc[pmt_indicies])
                        print("PMT Labels", pmt_labels)

                        # create a new figure
                        plt.scatter(x, y, s=1)

                        # plot the labels from the pmt_labels array on the same plot as text
                        for i, txt in enumerate(pmt_labels):
                            plt.annotate(pmt_labels[i], (x[i], y[i]))

                        # show the plot
                        plt.show()

                        window["-INFO-"].update(value=f'Auto labeled PMTs')

                    except Exception as e:
                        print(e)
                        window["-INFO-"].update(value=f'Failed to auto label PMTs')


                        # ## trying geopandas for creating a grid
                        # parse_data = [[item['Img'], item['ID'], item['SK'], item['X'], item['Y'], item['R'], item['theta'], item['Name']] for item in coord_dict]
                        # df = pd.DataFrame(coord_dict=parse_data, columns=['Img', 'ID', 'SK', 'X', 'Y', 'R', 'theta', 'Name'])
                        # pmt_plot = gdf.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.X, df.Y))
                        # pmt_plot.head()


                        ## place points inside a pseudo regular grid
                    #
                    #     nodesx = 100
                    #     sizex=20
                    #     sizey=20
                    #     firstx = min(coord_dict["X"])-20
                    #     firsty = min(coord_dict["Y"])-20
                    #
                    #     new, xt, yt = [], [], []
                    #     for i in point_coords:
                    #         xo = int((i[0]-firstx)/sizex) ## gives x grid cell size
                    #         yo = int((i[1]-firsty)/sizey)
                    #         new.append(nodesx*yo + xo)
                    #         xt.append(i[0])
                    #         yt.append(i[1])
                    #
                    #     sort_points = [x for (y,x) in sorted(zip(new, point_coords))]
                    #
                    #     ## index for where we have our identified PMT in sorted list
                    #     pmt_idx = np.where(np.array(point_coords) == first_pmt)[0][0]
                    #     print("The coordinates are apparently ", xt[pmt_idx], yt[pmt_idx])
                    #     print("Index where we labeled our PMT", pmt_idx)
                    #
                    #     # label_list = [0.0]*len(new)
                    #     # label_list[pmt_idx] = first_label
                    #
                    #     # for i, e in reversed(list(enumerate(label_list[0:pmt_idx]))):
                    #     #     if new[i]
                    #
                    #
                    #     plt.scatter(xt, yt)
                    #
                    #
                    #
                    #     for i in range(len(sort_points)):
                    #         plt.text(sort_points[i][0], sort_points[i][1], str(i))
                    #
                    #     plt.show()
                    #
                    #     ## meshgrid method
                    #     xv, yv = np.meshgrid(x,y,sparse=True)
                    #     # pmt_idx = np.argwhere((xv == first_pmt[0]) & (yv == first_pmt[1]))
                    #     # print("Grid index of identified point, ", pmt_idx)
                    #     fig = plt.figure()
                    #     ax = fig.add_subplot(111)
                    #     ax.plot(xv[0,:], yv[:,0], marker='o', color='k', linestyle='none')
                    #     for xy in zip(xv[0,:], yv[:,0]):
                    #         ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
                    #     ax.grid
                    #     # plt.plot(np.diag(xv), np.diag(yv), marker='o', color='k', linestyle='none')
                    #     plt.show()
                    #
                    # except Exception as e:
                    #     print(e)
                    #     print("Autolabeling did not work. Please try selecting a PMT again.")
                    #
                    #
        elif event.endswith('+UP'):
            
            made_point = False
            
            if values['-MOVE-']:
                
                try:
                    df_feature = func.move_feature(df, start_pt, end_pt, name)
                    window["-INFO-"].update(value=f"Moved point {df_feature['ID'].iloc[0]} from {start_pt} to {end_pt}")

                except Exception as e:
                    print(e)
                    
            elif values['-ERASE-']:
                try:
                    if len(figures)>1:
                        graph.delete_figure(fig)
                        df_erased_feature = func.del_point(df, start_pt)
                        window["-INFO-"].update(value=f"Erased point {df_erased_feature['ID'].iloc[0]}")

                except Exception as e:
                    print(e)
                    
            elif values['-MODIFY-']:
                try:
                    id_new = None
                    id_new = sg.popup_get_text('Please enter new ID (#####-## format)', title="Modifying Label")
                    if id_new:
                        df_modded_feature = func.modify_label(df, start_pt, id_new)
                        # Print the ID of the updated dataframe and the ID of the original dataframe
                        
                        window["-INFO-"].update(value=f"Updated ID from {df_modded_feature['ID'].iloc[0]} to index {df.loc[df_modded_feature.index, 'ID']}")

                except Exception as e:
                    print(e)

            func.erase_labels(graph, labels)                
            labels = func.plot_labels(graph, df, undistort)
                
            dragging = False

        elif event.endswith('+MOTION+'):
            window["-INFO-"].update(value=f"Mouse freely moving {values['-GRAPH-']}")

## =========================== Overlaying known coordinates =======================        

        if event == "-OVERLAY-": ## if we want to overlay known coordinates on an image
        
            print("Chose a points file to overlay on current image.")
            print(values["-OVERLAY-"])
            overlay_file = values["-OVERLAY-"]
            df = func.overlay_pts(overlay_file)
            print("Overlay dataframe", df)
            Img_ID = df['Img'].iloc[0]
            func.draw_pts(graph, df)
            print("Drawing points")
            func.erase_labels(graph, labels)                
            labels = func.plot_labels(graph, df, undistort)
            print("Plotting labels")

## ======================== Menubar functions =======================================        
        
        elif event == 'Copy':
            func.copy(window)
        
        elif event == '-SAVE-': ## saves annotated image and objects
            dir_name = os.path.dirname(filename)
            base = os.path.basename(filename)
            annotate_fname = os.path.join(str(dir_name), "annotated_"+str(base))
            # func.save_element_as_file(column, annotate_fname)
            
        elif event == '-PLOT_LABEL-':
            func.erase_labels(graph, labels)                
            labels = func.plot_labels(graph, df, undistort)
            
        elif event == '-ERASE_LABEL-':
            func.erase_labels(graph, labels)
        
        elif event == '-CSV-':
            folder = os.path.join(os.path.dirname(values["-FOLDER-"]),'Annotation_Coordinates')
            
            # Make folder if it doesn't exist
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            output_filepath_txt = os.path.join(folder, os.path.basename(os.path.splitext(filename)[0])+".txt")
            
            try:
                
                func.write_coords_to_file(df, output_filepath_txt)
                
            except Exception as e:
                print(e)
                sg.popup_ok("Could not save. Check terminal messages.")
            
        # elif event == 'Shift R':
            
        # This doesn't seem to do anything yet
        elif event == '&Undo point':
            
            try:
                graph.delete_figure(fig)

                # Delete the last row of dataframe
                df = df[:-1]
                
            except:
                pass
                    
        elif event == 'Change Canvas Size':
            scale = input("Please enter a scale factor '##': ")
            func.change_canvas_size(window, graph, scale)
            
        elif event == 'Resize Image':
            
            if ids is not None:
                # graph.delete_figure(ids) ## removes only the image
                graph.erase() ## removes all figures drawn on the graph
            
            # try:
            scale = input("Please enter a scale factor '##': ")
            im_array = np.array(image, dtype=np.uint8)
            data = func.resize(window, im_array, scale)
            func.change_canvas_size(window, graph, scale)
            graph.draw_image(data=data, location=(0,var.height))
            func.draw_pts(graph, df, undistort, scale=scale)
            
                
            # except:
            #     sg.popup_ok("Please check your size input format. E.g. ##, ##")
        
    window.refresh() ## refreshing the GUI to prevent it from hanging
    window.close() ## For when the user presses the Exit button
    
main()


# %%

import numpy as np

# load in data from .txt file "530.txt"
value_file = np.loadtxt("C:/Users\gaurr\OneDrive - TRIUMF\Super-K\Feature Detection & Labelling\TB3\BarrelSurveyRings\points/530.txt", usecols=[2,3])

# plot the X and Y coordinates from the third and fourth columns of the .txt file
import matplotlib.pyplot as plt

# use the second column to get the text labels for the associated points
labels = np.loadtxt("C:/Users\gaurr\OneDrive - TRIUMF\Super-K\Feature Detection & Labelling\TB3\BarrelSurveyRings\points/530.txt", usecols=[1], dtype=str)

# find indices the last characters of the labels elements are '00'
indices = [i for i, x in enumerate(labels) if x.endswith("00")]

# keep only 'indices' in the labels and value_file arrays
labels = labels[indices]
value_file = value_file[indices]

plt.scatter(value_file[:,0], 2750-value_file[:,1], s = 1)

# plot the text labels on the same plot but only using the fourth and fifth characters of the label
for i, txt in enumerate(labels):
    plt.annotate(labels[i][3:5], (value_file[i,0], 2750-value_file[i,1]))


# show the plot

plt.show()

