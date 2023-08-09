# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:24:03 2023

@author: gaurr091
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

import faulthandler
faulthandler.enable()

# %% Defining window objects and layout

## Main functiong
def main():

    window = func.make_main_window()
    location = 0
    graph = window["-GRAPH-"]

    name = None
    while name is None:
        # Prompt user for initials. If user clicks "Cancel", exit the program
        name = sg.popup_get_text('Please enter your initials:', 'Initials', default_text='')
        if name is None:
            sys.exit()

    Img_ID = None

    dragging = False
    made_point = False
    start_pt = end_pt = None
    ids = None ## used to specify which figure on sg.Graph to delete
    first_pmt = None
    pts_dir = None
    df = None

    labels = []
    points = []

    while True:

        event, values = window.read()
        ## 'event' is the key string of whichever element user interacts with
        ## 'values' contains Python dictionary that maps element key to a value

        if event == 'Exit' or event == sg.WIN_CLOSED: ## end loop if user clicks "Exit" or closes window
            break

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

                    points = func.draw_pts(graph, df)
                    labels = func.reload_plot_labels(graph, df, labels)

                    # show full image in second window
                    window2 = func.make_win2(df, filename)

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
                df = func.autoload_pts(pts_fname, name, var.mtx)
                Img_ID = df['Img'].iloc[0]

                points = func.draw_pts(graph, df)
                labels = func.reload_plot_labels(graph, df, labels)

                # show full image in second window
                window2 = func.make_win2(df, filename)

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
                        pmt_id = sg.popup_get_text('Please enter PMT ID #####', title="Adding PMT")
                        if pmt_id:
                            df = func.make_pmt(df, pmt_id, x, y, name)
                            graph.draw_point((x,y), color = 'red', size=8)

                    ## drawing bolt point
                    elif values["-BOLT_POINT-"]:

                        ## checks which pmt the bolt belongs to and returns ID of the PMT
                        ## along with the angle between the dynode and the bolt
                            df = func.make_bolt(df, x, y, name)
                            graph.draw_point((x,y), color = 'yellow', size=6)

                    labels = func.reload_plot_labels(graph, df, labels)

                    window["-INFO-"].update(value=f'Made point {df["ID"].iloc[-1]} at ({x}, {y})')

                    # saving the dataframe after a point has been made
                    output_filepath_txt = func.create_annotation_file(values, filename)
                    func.write_coords_to_file(df, output_filepath_txt)

                    window2 = func.make_win2(df, filename)
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
                    window2 = func.make_win2(df, filename)

                    # saving the dataframe after a move has been made
                    output_filepath_txt = func.create_annotation_file(values, filename)
                    func.write_coords_to_file(df, output_filepath_txt)

                except Exception as e:
                    print(e)

            elif values['-ERASE-']:
                try:
                    if len(figures)>1:

                        # ask user if they're sure that they want to delete the point in question
                        if sg.popup_yes_no('Are you sure you want to delete this point?', title="Deleting Point") == 'Yes':

                            graph.delete_figure(fig)
                            df_erased_feature, df_erased_bolts = func.del_point(df, start_pt)

                            # delete the bolts from df_erased_bolts from the graph
                            if df_erased_bolts is not None:
                                for index, row in df_erased_bolts.iterrows():
                                    # find the bolts in the graph
                                    for fig in graph.get_figures_at_location((row['X'], row['Y']))[1:]:
                                        graph.delete_figure(fig)
                                        break

                            window["-INFO-"].update(value=f"Erased point {df_erased_feature['ID'].iloc[0]}")
                            window2 = func.make_win2(df, filename)

                        elif sg.popup_yes_no('Are you sure you want to delete this point?', title="Deleting Point") == 'No':
                            pass

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
                        window2 = func.make_win2(df, filename)

                        # saving the dataframe after a modification has been made
                        output_filepath_txt = func.create_annotation_file(values, filename)
                        func.write_coords_to_file(df, output_filepath_txt)

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
                        label = int(sg.popup_get_text('Please enter label #####', title="Adding Label"))
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

                        print("Final dataframe with all labels. ", df.to_string())

                        func.autolabel_plot(df)

                        # final dataframe
                        df = func.finalize_df(df)

                        window["-INFO-"].update(value=f'Autolabeled PMTs')

                        first_pmt = None  ## to allow autolabeling again

                        window.refresh()
                        window["-COL-"].contents_changed()


                    except Exception as e:
                        print(e)
                        print("Auto labeling failed. Please select 'Autolabel' again.")
                        window["-INFO-"].update(value=f'Failed to auto label PMTs')

            labels = func.reload_plot_labels(graph, df, labels)

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

            points = func.draw_pts(graph, df)

            labels = func.reload_plot_labels(graph, df, labels)

## ======================== Menubar functions =======================================        
        
        elif event == 'Copy':
            func.copy(window)

        elif event == '-FILL_BOLTS-':
            df = func.fill_bolts(df, name)
            func.erase_pts(graph, points)
            points = func.draw_pts(graph, df)
            func.erase_labels(graph, labels)
            labels = func.plot_labels(graph, df)

        elif event == '-SAVE_IMAGE-': ## save the image with the points and labels
            output_filepath = func.save_image(values, window2, filename)
            
        elif event == '-PLOT_LABEL-':
            labels = func.reload_plot_labels(graph, df, labels)
            
        elif event == '-ERASE_LABEL-':
            func.erase_labels(graph, labels)
        
        elif event == '-CSV-':
            output_filepath_txt = func.create_annotation_file(values, filename)

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
            graph.draw_image(data=data, location=(0, var.height))
            func.draw_pts(graph, df, scale=scale)
            

            # except:
            #     sg.popup_ok("Please check your size input format. E.g. ##, ##")


    window.refresh() ## refreshing the GUI to prevent it from hanging
    window.close() ## For when the user presses the Exit button

main()