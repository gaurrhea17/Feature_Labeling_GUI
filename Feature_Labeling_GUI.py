# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:24:03 2023

@author: gaurr
"""

## Importing necessary packages

import numpy as np
import glob
from PIL import Image, ImageDraw
import io
import os.path
import PySimpleGUI as sg
import matplotlib
# %matplotlib tk ## will work as a command if file as .ipy extension
# export MPLBACKEND = TKAgg
import matplotlib.pyplot as plt
import cv2 as cv
# matplotlib.use("TkAgg") ##importing matplotlib with Tkinter rather than another framework like Qt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

#%%

# sg.theme_previewer() ## Use to see all the colour themes available
sg.theme('GreenTan') ## setting window colour scheme

## Converting JPG to PNG
def jpg_to_png(folder):
   
    for fname in folder: ## case sensitive
        if os.path.isfile(os.path.join(folder, fname)) and fname.lower().endswith((".JPG")):
            print("you found a picture")    
            # cv.imread(fname)
            # cv.imwrite(os.path.splitext(fname)[0]+".png")
        
path = "C:\\Users\\gaurr\\OneDrive - TRIUMF\\Hyper-K\\Calibration\\100MEDIA_OLDDrone\\100MEDIA_OLDDrone\\"
jpg_to_png(path)

## User selected a folder; need to get images in folder
def parse_folder(window, values):
    folder = values["-FOLDER-"]
    try:
        # To get the list of files in the folder
        file_list = os.listdir(folder)
    except:
        file_list = []
    
    fnames = [ ## filter down list of files to files ending with extension ".png" or ".gif"
        f
        for f in file_list
        ## case sensitive
        if os.path.isfile(os.path.join(folder, f)) 
        and f.lower().endswith((".gif", ".png"))
    ]
    window["-FILE LIST-"].update(fnames) ## list of files updated with chosen folder's contents

    return fnames
    
## Converting the pictures from array to data
def array_to_data(array):
    im = Image.fromarray(array)
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data
    
    
## User selected an image to display
def disp_image(window, values, fnames, location): 
    
    if location == 0:
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
    
        index = np.where(np.array(fnames) == values["-FILE LIST-"][0])[0][0]
    
    elif location == 1:
        index = np.where(np.array(fnames) == value_file)[0][0]
        filename = os.path.join(
            values["-FOLDER-"], fnames[index+1])
        values["-FILE LIST-"][0] = fnames[index+1]
        
    elif location == -1:
        index = np.where(np.array(fnames) == value_file)[0][0]
        filename = os.path.join(
            values["-FOLDER-"], fnames[index-1])
        values["-FILE LIST-"][0] = fnames[index-1]
    
    graph.draw_circle((200,150), 10, fill_color='black', line_color='white')
    window["-TOUT-"].update(filename) ## update the text with the filename selected
    # window["-IMAGE-"].update(filename=filename) ## update the image with the file selected
    im = Image.open(filename)
    im_array = np.array(im, dtype=np.uint8)
    data = array_to_data(im_array)
    graph.draw_image(data=data, location=(0,height))
    # graph.draw_circle((200,150), 10, fill_color='black', line_color='white')

    return filename, values["-FILE LIST-"][0]

## Annotating function
def draw_feature(event, values, window, dragging, start_pt, end_pt, prior_rect):

    global RED
    RED = 255,0,0
    global BLUE
    BLUE = 0,0,255
    
    x, y = values["-GRAPH-"]
    if not dragging:
        start_pt = (x, y)
        dragging = True
        drag_figures = graph.get_figures_at_location((x,y))
        lastxy = (x,y)
    else:
        end_pt = (x,y)
    if prior_rect:
        graph.delete_figure(prior_rect)
    
    delta_x, delta_y = x - lastxy[0], y - lastxy[1]
    lastxy = (x, y)
    
    if None not in (start_pt, end_pt):
        if values["-MOVE-"]:
            for fig in drag_figures:
                graph.move_figure(fig, delta_x, delta_y)
                graph.update()
        elif values["-POINT-"]:
            graph.draw_point((x,y), color = 'red', size=10)
        elif values['-ERASE-']:
            for figure in drag_figures:
                graph.delete_figure(figure)
        elif values['-CLEAR-']:
            graph.erase()
        elif values['-MOVEALL-']:
            graph.move(delta_x, delta_y)
    window["-INFO-"].update(value=f"Mouse {values['-GRAPH-']}")
    

def save_fig(img, filename):
    path = os.path.dirname(filename)
    base = os.path.basename(filename)
    cv.imwrite(path+base+"_annotated.txt",img)
    
#%% Defining window objects and layout

## Top menubar options
menu_butts = [['File', ['New', 'Open', 'Save', 'Exit', ]], ['Edit', ['Cut', 'Copy', 'Paste', 'Undo'], ],  ['Help', 'About...'], ]
menubar = [[sg.Menu(menu_butts)],]

## Window column 1: list of files in chosen folder

file_list_col = [
    [
     sg.Text("Image folder"),
     sg.In(size=(25,1), enable_events =True, key="-FOLDER-"),
     sg.FolderBrowse(),
     ],
    [sg.Listbox(values=[], enable_events=True, size=(40,20), key="-FILE LIST-")
     ], ## displays a list of paths to the images you can choose from to display
    ]


mov_col = [[sg.R("Erase item", 1, key="-ERASE-", enable_events = True)],
           [sg.R("Erase all", 1, key="-CLEAR-", enable_events = True)],
           [sg.R("Draw points", 1, key="-POINT-", enable_events = True)],
           [sg.R('Send to back', 1, key='-BACK-', enable_events=True)],
           [sg.R('Bring to front', 1, key='-FRONT-', enable_events=True)],
           [sg.R("Move everything", 1, key="-MOVEALL-", enable_events=True)],
           [sg.R("Move stuff", 1, key="-MOVE-", enable_events = True)],
           ]

## Window column 2: name of chosen image and image display
width, height = 400, 300

image_viewer_col_2 = [
    [sg.Text("Choose an image from list on the left: ")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Graph(canvas_size = (width, height), graph_bottom_left = (0,0), 
     graph_top_right = (width, height), key = "-GRAPH-", change_submits = True, ## allows mouse click events
     background_color = 'lightblue', enable_events = True, drag_submits = True, 
     right_click_menu=[[],['Erase Item',]]), sg.Col(mov_col, key="-COL-")],
    [sg.Text(key="-INFO-", size=(60,1))]]

post_process_col= [
    [sg.Button("Save Annotations", size = (15,1), key="-SAVE-")],
    [sg.Button("Reconstruct", size=(15,1), key="-RECON-")],
    ]

## Main function
def main():
    
    # ------ Full Window Layout
    layout= [
        [menubar, sg.Column(file_list_col), sg.VSeperator(), sg.Column(image_viewer_col_2), sg.VSeperator(), sg.Column(post_process_col),
         [sg.Button("Prev", size=(10,1), key="-PREV-"), sg.Button("Next", size=(10,1), key="-NEXT-")],]] 
    
    window = sg.Window("Image Labeling GUI", layout, resizable=True) ## putting together the user interface
    
    location = 0
    global value_file, graph
    
    graph = window["-GRAPH-"]
    dragging = False
    start_pt = end_pt = prior_rect = None

    while True:
        event, values = window.read()
        ## 'event' is the key string of whichever element user interacts with
        ## 'values' contains Python dictionary that maps element key to a value
        
        if event == "Exit" or event == sg.WIN_CLOSED: ## end loop if user clicks "Exit" or closes window
            break;
                 
        # Folder name filled in, we'll now make a list of files in the folder
        if event == "-FOLDER-": ## checking if the user has chosen a folder
            fnames = parse_folder(window, values)
                        
        elif event == "-FILE LIST-": ## User selected a file from the listbox
            if len(values["-FOLDER-"]) == 0 :
                select_folder = sg.popup_ok("Please select a folder first.")
            elif len(values["-FOLDER-"]) != 0:
                filename, value_file = disp_image(window, values, fnames, location=0)
        if event == "-NEXT-":
            filename, value_file = disp_image(window, values, fnames, location=1)
        if event == "-PREV-":
            filename, value_file = disp_image(window, values, fnames, location=-1)
            
            
## =============== Annotation Features ========================            
            
        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='fleur')          # not yet released method... coming soon!
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr') 
        
        if event == "-GRAPH-":
            draw_feature(event, values, window, dragging, start_pt, end_pt, prior_rect)
        
        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"Made point at {start_pt} to {end_pt}")
            start_pt, end_pt = None, None  # enable grabbing a new rect
            dragging = False
            prior_rect = None
        elif event.endswith('+RIGHT+'):  # Righ click
            window["-INFO-"].update(value=f"Right clicked location {values['-GRAPH-']}")
        elif event.endswith('+MOTION+'):  # Righ click
            window["-INFO-"].update(value=f"mouse freely moving {values['-GRAPH-']}")
        elif event == 'Erase item':
            if values['-GRAPH-'] != (None, None):
                drag_figures = graph.get_figures_at_location(values['-GRAPH-'])
                for figure in drag_figures:
                    graph.delete_figure(figure)
            
        # if event == "-SAVE-":
        #     save_fig()

    window.close() ## For when the user presses the Exit button
    
main()


#%% Trying extra programs

 ## binding mouse click to window image element
 
 # img = cv.imread(filename)
 # if img is None:
 #     print('Could not read image')
 # window["-IMAGE-"].bind("<Button-1>", draw_feature)
     
 # base = os.path.basename(filename) ##filename with extension
 # img_name = os.path.splitext(base)[0] ##filename without extension
     
 # cv.namedWindow(img_name)
 # cv.setMouseCallback(img_name, select_feature)

