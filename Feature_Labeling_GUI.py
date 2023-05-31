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

#%% Defining window objects and layout

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
           # [sg.R('Draw Rectangles', 1, key='-RECT-', enable_events=True)],
           # [sg.R('Draw Circle', 1, key='-CIRCLE-', enable_events=True)],
           # [sg.R('Draw Line', 1, key='-LINE-', enable_events=True)],
           [sg.R('Draw PMT points', 1,  key='-PMT_POINT-', enable_events=True)],
           [sg.R('Draw bolt points', 1,  key='-BOLT_POINT-', enable_events=True)],
           [sg.R('Erase item', 1, key='-ERASE-', enable_events=True)],
           [sg.R('Erase all', 1, key='-CLEAR-', enable_events=True)],
           [sg.R('Send to back', 1, key='-BACK-', enable_events=True)],
           [sg.R('Bring to front', 1, key='-FRONT-', enable_events=True)],
           [sg.R('Move Everything', 1, key='-MOVEALL-', enable_events=True)],
           [sg.R('Move Stuff', 1, key='-MOVE-', enable_events=True)],
           [sg.R("Auto-label", 1, key='-LABELING-', enable_events=True)],
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
    [sg.Button("Plot Labels", size = (15,1), key="-PLOT_LABEL-"), sg.Button("Remove Labels", size = (18,1), key="-ERASE_LABEL-")],
    [sg.Button("Write CSV", size = (15,1), key="-CSV-")],
    [sg.Button("Reconstruct", size=(15,1), key="-RECON-")],
    [sg.Button('Autolabel', size =(15,1), key='-AUTO_LABEL-')],
    [sg.Button('Zoom In'), sg.Button('Zoom Out')],
    [sg.Button("Shift R"), sg.Button("Shift L")],
    [
     sg.Text("Choose a file of overlay points: ")],
     [sg.In(size=(18,1), enable_events =True, key="-OVERLAY-")],
     [sg.FileBrowse()],
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
    
    name = input("Please enter your initials.")
    Img_ID = None
    
    dragging = False
    made_point = False
    start_pt = end_pt = None
    ids = None ## used to specify which figure on sg.Graph to delete
    pmt1 = None
    pts_dir = None
    df = None
    labels = []
    
    while True:
        event, values = window.read()
        ## 'event' is the key string of whichever element user interacts with
        ## 'values' contains Python dictionary that maps element key to a value
        
        if event == "Exit" or event == sg.WIN_CLOSED: ## end loop if user clicks "Exit" or closes window
            break;
                 
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
                    
                image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=0)
                pmt1 = None ## to allow autolabeling again
                window.refresh()
                window["-COL-"].contents_changed()
                
                ## Tries to overlay points if the image has associated coordinate file
                pts_dir = os.path.join(os.path.dirname(values["-FOLDER-"]), 'points')
                pts_file = os.path.basename(filename).split('.')[0]
                pts_fname = os.path.join(pts_dir, pts_file) + ".txt" 

                df = func.autoload_pts(pts_fname)
                Img_ID = df['Img'].iloc[0]
                func.draw_pts(graph, df)
                func.erase_labels(graph, labels)                
                labels = func.plot_labels(graph, df)

                                        
        if event == "-NEXT-" or event == "-PREV-":
            
            if ids is not None:
                # graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                graph.erase()
            
            location=1
            if event == "-PREV-":
                location=-1

            image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=location)
            pmt1 = None ##to allow autolabeling again
            window.refresh()
            window["-COL-"].contents_changed()
            
            pts_file = os.path.basename(filename).split('.')[0]
            pts_fname = os.path.join(pts_dir, pts_file) + ".txt"             
            df = func.autoload_pts(values, graph, pts_fname)
            Img_ID = df['Img'].iloc[0]
            func.draw_pts(graph, df)
        
## =============== Annotation Features ========================            
            
        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='cross')
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr') 
        
        if event == "-GRAPH-":

            x, y = values["-GRAPH-"]
            
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
                
                ## drawing PMT point
                if values["-PMT_POINT-"]:                    
                    pmt_id = sg.popup_get_text('Please enter PMT ID', title="Adding PMT")
                    df = func.make_pmt(df, pmt_id, x, y, name)
                    graph.draw_point((x,y), color = 'red', size=8)

                ## drawing bolt point
                elif values["-BOLT_POINT-"]:

                    ## checks which pmt the bolt belongs to and returns ID of the PMT
                    ## along with the angle between the dynode and the bolt    
                    try:
                        df = func.make_bolt(df, x, y, name)
                        graph.draw_point((x,y), color = 'yellow', size=8)
                    
                    except Exception as e:
                        print(e)
                        print("Your last point could not be added. Please try again.")

                window["-INFO-"].update(value=f'Made point at ({x}, {y})')
                made_point = True
                
    ## ================== AUTO LABELING ===============================
            
            elif values["-LABELING-"]:
                
                if pmt1 == None:
                    sg.popup_ok("You selected your first PMT.")
                
                    try:
                        first_pmt = graph.get_figures_at_location(values['-GRAPH-'])
                        for fig in first_pmt[1:]:
                            pmt1 = func.get_marker_center(graph, fig)
                            print("Coordinate of identified PMT", pmt1)
                            break;
                    
                        for i in range(len(coord_dict["ID"])):
                            # print("Looking for PMT coordinates so you can enter an ID.")
                            if round(coord_dict["X"][i],1) == pmt1[0] and round(coord_dict["Y"][i],1) == pmt1[1]:
                                    first_label = input("Please enter this PMT's ID.")
                                    coord_dict["SK"][i] = first_label
                                    index = i ##index in full dictionary of identified PMT
                                    print(f"Index of your identified PMT is {index} in the dictionary.")
                                    break;
                            else:
                                continue;
                    
                        x, y = [], [] ## making lists of the PMT points
                        for i in range(len(coord_dict["ID"])):
                            if coord_dict["ID"][i].endswith('00'):
                                x.append(coord_dict["X"][i])
                                y.append(coord_dict["Y"][i])
                        
                        # create list of (x,y) coordinates for PMTs in your picture
                        point_coords = list(zip(x, y))
                        
                        # ## trying geopandas for creating a grid
                        # parse_data = [[item['Img'], item['ID'], item['SK'], item['X'], item['Y'], item['R'], item['theta'], item['Name']] for item in coord_dict]
                        # df = pd.DataFrame(coord_dict=parse_data, columns=['Img', 'ID', 'SK', 'X', 'Y', 'R', 'theta', 'Name'])
                        # pmt_plot = gdf.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.X, df.Y))
                        # pmt_plot.head()
                        
                        
                        ## place points inside a pseudo regular grid
                        
                        nodesx = 100
                        sizex=20
                        sizey=20
                        firstx = min(coord_dict["X"])-20
                        firsty = min(coord_dict["Y"])-20
                        
                        new, xt, yt = [], [], []
                        for i in point_coords:
                            xo = int((i[0]-firstx)/sizex) ## gives x grid cell size
                            yo = int((i[1]-firsty)/sizey)
                            new.append(nodesx*yo + xo)
                            xt.append(i[0])
                            yt.append(i[1])
                        
                        sort_points = [x for (y,x) in sorted(zip(new, point_coords))]
                        
                        ## index for where we have our identified PMT in sorted list
                        pmt_idx = np.where(np.array(point_coords) == pmt1)[0][0]
                        print("The coordinates are apparently ", xt[pmt_idx], yt[pmt_idx])
                        print("Index where we labeled our PMT", pmt_idx)
                        
                        # label_list = [0.0]*len(new)
                        # label_list[pmt_idx] = first_label
                        
                        # for i, e in reversed(list(enumerate(label_list[0:pmt_idx]))):
                        #     if new[i]
                

                        plt.scatter(xt, yt)
                        
                        
                        
                        for i in range(len(sort_points)):
                            plt.text(sort_points[i][0], sort_points[i][1], str(i))
                        
                        plt.show()

                        ## meshgrid method
                        xv, yv = np.meshgrid(x,y,sparse=True)
                        # pmt_idx = np.argwhere((xv == pmt1[0]) & (yv == pmt1[1]))
                        # print("Grid index of identified point, ", pmt_idx)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.plot(xv[0,:], yv[:,0], marker='o', color='k', linestyle='none')
                        for xy in zip(xv[0,:], yv[:,0]):
                            ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
                        ax.grid
                        # plt.plot(np.diag(xv), np.diag(yv), marker='o', color='k', linestyle='none')
                        plt.show()
                    
                    except Exception as e:
                        print(e)
                        print("Autolabeling did not work. Please try selecting a PMT again.")
        
        
        elif event.endswith('+UP'):
            
            made_point = False
            
            if values['-MOVE-']:
                
                try:
                    print(start_pt, end_pt)
                    df = func.move_feature(df, start_pt, end_pt, name)
                    window["-INFO-"].update(value=f"Moved point from {start_pt} to {end_pt}")

                except Exception as e:
                    print(e)
                    
            elif values['-ERASE-']:
                try:
                    func.del_point(df, start_pt[0], start_pt[1])
                    graph.delete_figure(fig)

                except Exception as e:
                    print(e)
                    
            func.erase_labels(graph, labels)                
            labels = func.plot_labels(graph, df)
                
            dragging = False

        elif event.endswith('+MOTION+'):
            window["-INFO-"].update(value=f"Mouse freely moving {values['-GRAPH-']}")

## =========================== Overlaying known coordinates =======================        

        if event == "-OVERLAY-": ## if we want to overlay known coordinates on an image

            print(values["-OVERLAY-"])
            # This needs to be re-written following the same method as the above func.autoload_pts

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
            labels = func.plot_labels(graph, df)
            
        elif event == '-ERASE_LABEL-':
            func.erase_labels(graph, labels)
        
        elif event == '-CSV-':
            folder = os.path.join(os.path.dirname(values["-FOLDER-"]),'Annotation_Coordinates')
            output_filepath = os.path.join(folder, os.path.basename(os.path.splitext(filename)[0])+".txt")

            try:
                func.write_coords_to_csv(df, output_filepath)
                
            except Exception as e:
                print(e)
                print("Did not save. Check that the file is not open.")

            
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
            func.draw_pts(graph, df, scale)
            
                
            # except:
            #     sg.popup_ok("Please check your size input format. E.g. ##, ##")
        
    window.refresh() ## refreshing the GUI to prevent it from hanging
    window.close() ## For when the user presses the Exit button
    
main()
