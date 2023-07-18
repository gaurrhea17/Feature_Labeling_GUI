# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:24:03 2023

@author: gaurr
"""

## Importing necessary packages
import sys

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

import Feature_Labeling_Functions as func
import Feature_Labeling_Variables as var


# %% Defining window objects and layout

# sg.theme_previewer() ## Use to see all the colour themes available
sg.theme('LightPurple') ## setting window colour scheme
# sg.set_options(dpi_awareness=True) ## setting window options

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
    # [sg.Button("Save Annotations", size = (15,1), key="-SAVE-")], ## was supposed to save image with points drawn on it that could be loaded into GUI
    [sg.Button("Plot Labels", size = (15,1), key="-PLOT_LABEL-"), sg.Button("Remove Labels", size = (18,1), key="-ERASE_LABEL-")],
    [sg.Button("Write CSV", size = (15,1), key="-CSV-")],
    # [sg.Button("Reconstruct", size=(15,1), key="-RECON-")], # was supposed to open panel to perform real-time reconstruction
    [sg.Button('Autolabel', size =(15,1), key='-AUTO_LABEL-')],
    # [sg.Button('Zoom In'), sg.Button('Zoom Out')],
    # [sg.Button("Shift R"), sg.Button("Shift L")], # was supposed to shift image left or right
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
                first_pmt = None ## to allow autolabeling again
                window.refresh()
                window["-COL-"].contents_changed()

                try:
                    ## Tries to overlay points if the image has associated coordinate file
                    pts_dir = os.path.join(os.path.dirname(values["-FOLDER-"]), 'points')
                    pts_file = os.path.basename(filename).split('.')[0]
                    pts_fname = os.path.join(pts_dir, pts_file) + ".txt"

                    df = func.autoload_pts(pts_fname, name, var.mtx)
                    print("Autoloaded the points. The new dataframe: \n", df)
                    Img_ID = df['Img'].iloc[0]
                    func.draw_pts(graph, df)
                    print("Drawing the points")
                    func.erase_labels(graph, labels)
                    labels = func.plot_labels(graph, df)
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

            image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=location)
            first_pmt = None ##to allow autolabeling again
            window.refresh()
            window["-COL-"].contents_changed()

            try:
                pts_file = os.path.basename(filename).split('.')[0]
                pts_fname = os.path.join(pts_dir, pts_file) + ".txt"
                df = func.autoload_pts(pts_fname, name)
                Img_ID = df['Img'].iloc[0]
                func.draw_pts(graph, df)
                func.erase_labels(graph, labels)
                labels = func.plot_labels(graph, df)
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
                        df_modded_feature = func.modify_label(df, start_pt, id_new, name)
                        # Print the ID of the updated dataframe and the ID of the original dataframe

                        window["-INFO-"].update(value=f"Updated ID from {df_modded_feature['ID'].iloc[0]} to index {df.loc[df_modded_feature.index, 'ID']}")

                except Exception as e:
                    print(e)

            ## User clicks on a PMT and program gets the PMTs coordinates from the graph.
            elif values["-LABELING-"]:

                if first_pmt == None:

                    try:
                        # get figure at location
                        fig = graph.get_figures_at_location(values['-GRAPH-'])[1]
                        curr_x, curr_y = func.get_marker_center(graph, fig)
                        print("Coordinates of selected PMT: ", curr_x, curr_y)

                        # look for curr_x and curr_y in the dataframe
                        curr_pos = (curr_x, curr_y)
                        df_feature = func.get_current_feature(df, curr_pos)
                        first_pmt = df_feature['ID'].iloc[0]
                        print("ID and coordinates of selected PMT: ", first_pmt, curr_pos)

                        # insert column with name 'Labels' at the end of the dataframe
                        df.insert(len(df.columns), 'Labels', None)
                        print("Made new column in dataframe.")

                        # get the index of the first pmt and get user input for this PMT's label. Put input into df['Labels'] column at index
                        index = df[df['ID'] == first_pmt].index[0]
                        label = int(sg.popup_get_text('Please enter label', title="Adding Label"))
                        df.at[index, 'Labels'] = label
                        print("Inserted label into dataframe.")

                        # finds PMT in dataframe and the other PMTs in the same row and column. Labels the row and column PMTs.
                        df, new_ref, lesser_x_row, greater_x_row, lesser_y_col, greater_y_col, row1, column1 = func.autolabel(df, first_pmt, label)

                        col_len = len(column1)
                        row_len = len(row1)

                        count = 0
                        while count < col_len or count < row_len:
                            if count < col_len:
                                new_ref, new_label = func.get_next_ref(df, count, column1)

                                # autolabeling PMTs in the same row as new_ref
                                df, new_ref, lesser_x_row, greater_x_row, lesser_y_col, greater_y_col, row, column = func.autolabel(df, new_ref, new_label)

                            if count < row_len:
                                new_ref2, new_label2 = func.get_next_ref(df, count, row1)

                                # autolabeling PMTs in the same column as new_ref2
                                df, new_ref2, lesser_x_row2, greater_x_row2, lesser_y_col2, greater_y_col2, row2, column2 = func.autolabel(df, new_ref2, new_label2)

                            count += 1



                        # print("Final dataframe with all labels. ", df.to_string())

                        func.autolabel_plot(df)

                        # final dataframe
                        df = func.finalize_df(df)

                        window["-INFO-"].update(value=f'Auto labeled PMTs')

                        first_pmt = None  ## to allow autolabeling again

                        window.refresh()
                        window["-COL-"].contents_changed()


                    except Exception as e:
                        print(e)
                        print("Auto labeling failed. Please select 'Autolabel' again.")
                        window["-INFO-"].update(value=f'Failed to auto label PMTs')

            func.erase_labels(graph, labels)
            labels = func.plot_labels(graph, df)

            # unselect 'Autolabel' radio button allow autolabeling again
            window['-LABELING-'].update(value=False, visible=True)
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
            labels = func.plot_labels(graph, df)
            print("Plotting labels")

## ======================== Menubar functions =======================================        
        
        elif event == 'Copy':
            func.copy(window)
        
        # elif event == '-SAVE-': ## saves annotated image and objects
        #     dir_name = os.path.dirname(filename)
        #     base = os.path.basename(filename)
        #     annotate_fname = os.path.join(str(dir_name), "annotated_"+str(base))
            # func.save_element_as_file(column, annotate_fname)
            
        elif event == '-PLOT_LABEL-':
            func.erase_labels(graph, labels)                
            labels = func.plot_labels(graph, df)
            
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
            func.draw_pts(graph, df, scale=scale)
            
                
            # except:
            #     sg.popup_ok("Please check your size input format. E.g. ##, ##")
        
    window.refresh() ## refreshing the GUI to prevent it from hanging
    window.close() ## For when the user presses the Exit button
    
main()

#%%

