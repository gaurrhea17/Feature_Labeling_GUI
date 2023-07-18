# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:08:07 2023

This file hosts all of the user-defined functions referenced in the
"Feature_Labeling_GUI.py" code. 

@author: gaurr
"""

## Importing necessary packages
import sys
import json
import math
import numpy as np
import glob
from PIL import Image, ImageDraw, ImageGrab
import io
import base64
import csv
import pandas as pd
import os.path
import PySimpleGUI as sg
import matplotlib
# %matplotlib tk ## will work as a command if file as .ipy extension
# export MPLBACKEND = TKAgg
import matplotlib.pyplot as plt
import cv2 as cv


import Feature_Labeling_Variables as var
import Feature_Labeling_Diagnostic_Plot as diag_plt

# %%


img_types = (".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG")


## Converting a single JPG to PNG
def jpg_to_png(jpg_path):
    """ This function can be called to convert .jpg images to .png format.
   Args:
       jpg_path (str): The path of the .jpg file to be converted specified with the extension."""

    # Open the .jpg file using PIL's Image module
    with Image.open(jpg_path) as img:
        # Convert the image to RGBA mode (necessary for saving as .png format)
        img_rgba = img.convert("RGBA")

        # Get the file name and extension of the .jpg file
        filename = os.path.splitext(os.path.basename(jpg_path))[0]

        # Save the converted image as .png format to the specified directory
        png_dir = os.path.dirname(jpg_path)
        png_path = os.path.join(png_dir, filename) + ".png"
        img_rgba.save(png_path, format="png")

        return png_path


## Converting a folder of JPGs to PNGs

def jpg_folder_to_png(directory):
    for fname in os.listdir(directory):
        if fname.endswith(".jpg"):
            im = Image.open(os.path.join(directory, fname))
            img_rgba = im.convert('RGBA')

            file = os.path.splitext(fname)[0]
            new_filename = str(os.path.join(directory, file)) + '.png'
            img_rgba.save(new_filename)

            os.remove(os.path.join(directory, fname))  ## deletes the .jpg
            continue
        else:
            continue


def change_canvas_size(window, graph, scale):
    """" This function allows the user to input the size of the graph/column/scrollable area."""
    scale = int(scale)
    graph.Widget.configure(width=var.width * scale, height=var.height * scale)
    window.refresh()
    window["-COL-"].contents_changed()


def parse_folder(window, values):
    """ Called when the user selects a folder. This function finds all of the files with
    .png or .gif extensions in the folder and adds them to the listbox."""

    folder = values["-FOLDER-"]
    try:
        # To get the list of files in the folder
        file_list = os.listdir(folder)
    except:
        file_list = []

    fnames = [  ## filter down list of files to files ending with extensions from img_types
        f
        for f in file_list
        ## case sensitive
        if os.path.isfile(os.path.join(folder, f)) and
           f.lower().endswith(img_types)
    ]
    fnames.sort()
    window["-FILE LIST-"].update(fnames)  ## list of files updated with chosen folder's contents

    return fnames


def array_to_data(im, fname):
    """ Converting images from array type to bytes to display in the graph element."""

    # im = Image.fromarray(array)

    with io.BytesIO() as output:
        if im.format == "PNG":
            im.save(output, format="PNG")
        else:
            png_img = jpg_to_png(fname)
            png_img = Image.open(png_img)
            png_img.save(output, format="PNG")
        data = output.getvalue()

    return data


def get_curr_coords(graph, drag_figures):
    for fig in drag_figures[1:]:
        current_coords = graph.get_bounding_box(fig)
        curr_x = (current_coords[0][0] + current_coords[1][0]) / 2
        curr_y = (current_coords[0][1] + current_coords[1][1]) / 2
        point = curr_x, curr_y
        return curr_x, curr_y, point


def resize(window, array, scale):
    """ To allow the user to resize the image displayed in the graph."""

    im = Image.fromarray(array)
    w, h, c = np.array(array).shape  ## width, height, channel
    print(w, h)
    scale = int(scale)
    im = im.resize((w * scale, h * scale), Image.ANTIALIAS)
    print(w * scale, h * scale)

    with io.BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()

    # window.refresh()
    # window["-COL-"].contents_changed()
    return data


global filename


## input intrinsic parameters and build camera matrix and distortion parameter array

def build_camera_matrix(focal_length, principle_point, skew):
    return np.array([
        [focal_length[0], skew, principle_point[0]],
        [0, focal_length[1], principle_point[1]],
        [0, 0, 1]], dtype=float)


def build_distortion_array(radial_distortion, tangential_distortion):
    if radial_distortion.shape[0] > 2:
        return np.concatenate((radial_distortion[:2], tangential_distortion, radial_distortion[2:])).reshape((-1, 1))
    else:
        return np.concatenate((radial_distortion, tangential_distortion)).reshape((4, 1))


def camera_model(fx, fy, cx, cy, k1, k2, p1, p2, skew):
    focal_length = np.array([fx, fy])
    principle_point = np.array([cx, cy])
    radial_distortion = np.array([k1, k2])
    tangential_distortion = np.array([p1, p2])
    mtx = build_camera_matrix(focal_length, principle_point, skew)
    dist = build_distortion_array(radial_distortion, tangential_distortion)
    return focal_length, principle_point, radial_distortion, tangential_distortion, mtx, dist


## User selected an image to display
def disp_image(window, values, fnames, location):
    """ This function is called when the user selects an image they wish to display.
    This can be either when an image is selected from the listbox or the "Next" 
    or "Previous" buttons are selected. 
    
    The graph element is then updated with the image and the filename is also updated. """

    if location == 0:
        fname = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )

        index = np.where(np.array(fnames) == values["-FILE LIST-"][0])[0][0]
        print("Image name: ", fname)

    elif location == 1:

        index = np.where(np.array(fnames) == os.path.basename(window["-TOUT-"].get()))[0][0]
        fname = os.path.join(
            values["-FOLDER-"], fnames[index + 1])
        values["-FILE LIST-"][0] = fnames[index + 1]

    elif location == -1:
        index = np.where(np.array(fnames) == os.path.basename(window["-TOUT-"].get()))[0][0]
        fname = os.path.join(
            values["-FOLDER-"], fnames[index - 1])
        values["-FILE LIST-"][0] = fnames[index - 1]

    window["-TOUT-"].update(fname)  ## update the text with the filename selected

    ## converting image type to array and then array to data
    im = Image.open(fname)

    ## Converting image to array to data -- for display of distorted images
    data = array_to_data(im, fname)

    ids = window["-GRAPH-"].draw_image(data=data, location=(0, var.height))

    with open(fname, 'rb') as f:
        im_bytes = f.read()
    pil_image = Image.open(io.BytesIO(im_bytes))

    return im, pil_image, fname, ids


def copy(window):
    """"This function can be used to copy the annotated image onto the
    user's clipboard.
    
    DOES NOT CURRENTLY WORK"""

    widget = window.find_element_with_focus().widget
    if widget.select_present():
        text = widget.selection_get()
        window.TKroot.clipboard_clear()
        window.TKroot.clipboard_append(text)


def save_element_as_file(element, fname):
    """
    Saves any element as an image file.  Element needs to have an underlying Widget available (almost if not all of
    them do) :param element: The element to save :param filename: The filename to save to. The extension of the
    filename determines the format (jpg, png, gif)
    """
    try:
        widget = element.Widget
        # box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(),
        # widget.winfo_rooty() + widget.winfo_height())
        box = (0, 0, 500, 500)
        grab = ImageGrab.grab(bbox=box)
        grab.save(fname)
        sg.popup_ok("Your image has been saved with annotations!")
    except Exception as e:
        sg.popup_ok("The file could not be saved.")
        print(e)


def draw_pts(graph, df, scale=1):
    for index, row in df.iterrows():

        draw_coords = (float(row[2]) * scale, float(row[3]) * scale)
        color = 'yellow'
        size = 6
        if str(row[1]).endswith('00'):  # PMT
            color = 'red'
            size = 8

        graph.draw_point(draw_coords, color=color, size=size)


def autoload_pts(fname, name, mtx):
    print("Loading points file:", fname)

    ## Read space delimited point file into dataframe
    df = pd.read_csv(fname, delim_whitespace=True, names=["Img", "ID", "X", "Y", "Name"])

    ## Remove points not corresponding to PMT/dynode center nor bolts (end in ## > 24)
    df = df[df['ID'].apply(lambda id: float(id[-2:]) <= var.NBOLTS)]

    ## Add columns for automatic bolt labeling variables
    df['R'] = np.nan
    df['theta'] = np.nan

    ## Invert Y coordinate from the FeatureReco convention
    if var.invert_y:
        df['Y'] = df['Y'].map(lambda Y: var.height - Y)


    ## Process all the points
    for index, row in df.iterrows():

        ## PMT (dynode) centers
        if str(row[1]).endswith('00'):
            pass

        ## Bolts
        else:
            ## Get PMT ID associated to this bolt                
            PMT_ID = row[1][:-2] + '00'
            df_pmt = df.loc[df['ID'] == PMT_ID]

            if len(df_pmt.index) > 1:
                sys.exit("Unexpected duplicate PMT_ID " + PMT_ID)

            ## Calculate parameters needed for automatic bolt labeling
            pmt_x = float(df_pmt['X'].iloc[0])
            pmt_y = float(df_pmt['Y'].iloc[0])

            df.at[index, 'R'] = np.sqrt((float(row[2]) - pmt_x) ** 2 + (float(row[3]) - pmt_y) ** 2)
            df.at[index, 'theta'] = angle_to((float(row[2]), float(row[3])), (pmt_x, pmt_y))

    df = duplicate_check(df)
    return df


def overlay_pts(fname):
    ## Read space delimited point file into dataframe
    df = pd.read_csv(fname, delim_whitespace=True,
                     names=["Img", "ID", "X", "Y", "Undistort_X", "Undistort_Y", "Name", "R", "theta"])
    return df

def get_current_feature(df, position):
    # finds clicked point in dataframe and matches ID to feature
    df_feature = df[(round(df['X']) == round(position[0])) & (round(df['Y']) == round(position[1]))]

    if len(df_feature.index) == 0:
        print("No feature found at this location!")
        raise

    return df_feature

def recalculate_one_bolt(df, df_feature):
    
    feature_ID = df.at[df_feature.index[0], 'ID']
    
    ## Get PMT ID associated to this bolt
    PMT_ID = feature_ID[:-2] + '00'
    df_pmt = df.loc[df['ID'] == PMT_ID]

    if len(df_pmt.index) == 0:
        print("No PMT found with ID " + PMT_ID)
        raise

    elif len(df_pmt.index) > 1:
        print("Unexpected duplicate PMT_ID " + PMT_ID)
        raise

    bolt_x = float(df_feature['X'].iloc[0])
    bolt_y = float(df_feature['Y'].iloc[0])
    pmt_x = float(df_pmt['X'].iloc[0])
    pmt_y = float(df_pmt['Y'].iloc[0])
    bolt_r, theta = calc_bolt_properties(bolt_x, bolt_y, pmt_x, pmt_y)

    df.loc[df_feature.index, 'R'] = bolt_r
    df.loc[df_feature.index, 'theta'] = theta

def modify_label(df, position, id_new, name):
    df_feature = get_current_feature(df, position)

    # get the 'ID' entry from df_feature
    feature_ID = df.at[df_feature.index[0], 'ID']

    # Directly modify ID of feature in original dataframe
    df.loc[df_feature.index, 'ID'] = id_new

    # Relabel all bolts associated to this PMT
    if str(id_new).endswith('00'):
        df_bolts = df[df['ID'].apply(lambda id: str(id)[:-2] == str(feature_ID)[:-2] and not id.endswith('00'))]

        # replace the first 5 characters of all bolts with the first 5 characters of id_new
        df.loc[df_bolts.index, 'ID'] = df_bolts['ID'].apply(lambda id: str(id_new)[:-2] + str(id)[-2:])

    # Update bolt_r and theta for this feature if it is a bolt
    if not str(id_new).endswith('00'):
        recalculate_one_bolt(df, df_feature)

    # update name for this feature
    df.loc[df_feature.index, 'Name'] = name

    return df_feature

def del_point(df, position):
    df_feature = get_current_feature(df, position)

    df.drop(df_feature.index, inplace=True)

    df.reset_index(drop=True, inplace=True)  ## resetting indices after deleting point

    return df_feature


def safe_open_w(path):
    """ Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w+', newline='')


def reverseEnum(data: list):
    for i in range(len(data) - 1, -1, -1):
        yield i, data[i]


def duplicate_check(df):

    # find indicies of all duplicate rows according to the 'ID' column
    duplicateRowsDF = df[df.duplicated(['ID'])]

    # drop duplicate rows
    df.drop_duplicates(subset=df.columns[1], keep="first", inplace=True)  ## removing duplicate based on ID and keeping the first entry

    # reset indices after removing duplicates
    df.reset_index(drop=True, inplace=True)

    # print which IDs were removed and their associated 'X' and 'Y' coordinates
    for index, row in duplicateRowsDF.iterrows():
        print("Duplicate ID: ", row['ID'], " at (", str(row['X']), ", " + str(row['Y']) + ") will be removed.")

    return df


def write_coords_to_file(df, filename):
    # df = duplicate_check(df)
    df = df.sort_values(by=['ID'])

    # Write X/Y column with precision 
    df['X'] = df['X'].map(lambda x: '{:.1f}'.format(x))
    df['Y'] = df['Y'].map(lambda y: '{:.1f}'.format(y))

    headers = ["Img", "ID", "X", "Y", "Name"]
    df[headers].to_csv(filename, index=False, header=False, sep='\t', na_rep='NULL')
    print("Saved .txt annotations: ", filename)

def angle_to(p1, p2, rotation=270, clockwise=False):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) - rotation
    if not clockwise:
        angle = -angle
    return angle % 360


def get_closest_pmt(df, x, y):
    df_pmts = df[df['ID'].apply(lambda feature_id: float(feature_id[-2:]) == 0)]
    df_closest = df_pmts.iloc[((df_pmts['X'] - x) ** 2 + (df_pmts['Y'] - y) ** 2).argsort()[:1]]
    pmt_id = df_closest['ID'].iloc[0][:5]
    pmt_x = df_closest['X'].iloc[0]
    pmt_y = df_closest['Y'].iloc[0]

    return pmt_id, pmt_x, pmt_y

def calc_bolt_properties(bolt_x, bolt_y, pmt_x, pmt_y):
    bolt_to_pmt = (bolt_x - pmt_x, bolt_y - pmt_y)
    bolt_r = np.sqrt(bolt_to_pmt[0] ** 2 + bolt_to_pmt[1] ** 2)
    theta = angle_to((bolt_x, bolt_y), (pmt_x, pmt_y))

    return bolt_r, theta

def make_bolt(df, bolt_x, bolt_y, name):
    ## Find which PMT the bolt is closest to
    ### (WARNING: this will not work if the bolt is closer to a different PMT center))

    pmt_id, pmt_x, pmt_y = get_closest_pmt(df, bolt_x, bolt_y)
    # print("PMT number where min. distance between bolt and dynode :", pmt_id)

    ## Get list of existing bolts for this PMT
    df_bolts = df[df['ID'].apply(lambda feature_id: (feature_id[:5] == pmt_id) & (float(feature_id[-2:]) > 0))]
    # print("The number of bolts for this PMT is ", len(df_bolts.index))

    ## Don't add more than 24 bolts
    if len(df_bolts.index) >= var.NBOLTS:
        raise Exception("Already reached max number of bolts for this PMT! Erase a bolt first.")

    ## calculate angle between PMT center and bolt
    bolt_r, theta = calc_bolt_properties(bolt_x, bolt_y, pmt_x, pmt_y)
    # print(f"Angle between PMT and bolt {theta}")

    # Get entry in df_bolts with value closest to 'theta'
    df_closest_theta = df_bolts.iloc[(abs((theta - df_bolts['theta'] + 180) % 360 - 180)).argsort()[:1]]
    # print("Closest theta is ", df_closest_theta)

    # Assign 'ID' +1 if theta is greater than the closest theta or 'ID' -1 if theta is less than the closest theta
    if len(df_closest_theta.index) == 0:
        # print("No bolts for this PMT yet!")
        bolt_label = '-01'

    else:
        closest_bolt_id = int(df_closest_theta['ID'].iloc[0][-2:])
        closest_theta = df_closest_theta['theta'].iloc[0]

        # Determine if theta is clockwise to closest_theta
        if (theta - closest_theta - 180) % 360 - 180 > 0:  # clockwise

            if closest_bolt_id == var.NBOLTS:
                bolt_label = '-01'

            else:
                bolt_label = '-{:02d}'.format(closest_bolt_id + 1)

        # Counter-clockwise
        else:

            if closest_bolt_id == 1:
                bolt_label = '-{:02d}'.format(var.NBOLTS)

            else:
                bolt_label = '-{:02d}'.format(closest_bolt_id - 1)

    New_ID = pmt_id.zfill(5) + bolt_label

    check_existing_id(df, New_ID)

    # append new row to dataframe
    df_new_bolt = pd.DataFrame(
        {'Img': df['Img'].iloc[0], 'ID': New_ID, 'X': bolt_x, 'Y': bolt_y, 'Name': name, 'R': bolt_r, 'theta': theta},
        index=[0])
    df = pd.concat([df, df_new_bolt], ignore_index=True)
    # df.loc[len(df.index)] = [df['Img'].iloc[0], New_ID, bolt_x, bolt_y, name, bolt_r, theta]

    print("Inserted your new bolt", df.tail(1))

    return df


def make_pmt(df, pmt_id, pmt_x, pmt_y, name):

    New_ID = pmt_id.zfill(5) + "-00"

    check_existing_id(df, New_ID)

    df_new_pmt = pd.DataFrame(
        {'Img': df['Img'].iloc[0], 'ID': New_ID, 'X': pmt_x, 'Y': pmt_y, 'Name': name, 'R': np.nan,
         'theta': np.nan}, index=[0])
    df = pd.concat([df, df_new_pmt], ignore_index=True)

    # df = df.sort_values(by=['ID'])
    print("Inserted your new PMT", df.tail(1))

    # recalculate all bolt properties for this PMT
    recalculate_all_bolts(df, New_ID, pmt_x, pmt_y)

    return df

def recalculate_all_bolts(df, New_ID, pmt_x, pmt_y):

    # find all entries where df['ID'] starts with New_ID[:-2] and ends not with '00' and print df['ID'] inside lambda function
    df_bolts = df[df['ID'].apply(lambda feature_id: (feature_id != None) and (feature_id[:-2] == New_ID[:-2]) and (feature_id[-2:] != '00'))]

    # Modify R and theta in df_bolts
    for index, row in df_bolts.iterrows():
        bolt_r, theta = calc_bolt_properties(row['X'], row['Y'], pmt_x, pmt_y)
        df.loc[index, 'R'] = bolt_r
        df.loc[index, 'theta'] = theta

def check_existing_id(df, new_id):
    if new_id in df['ID'].values:
        print ("ID already exists:", new_id)
        raise Exception("ID already exists! Please check existing bolt labels or choose a different PMT ID.")

def move_feature(df, start_pt, end_pt, name):

    df_feature = get_current_feature(df, start_pt)

    # Modify X and Y in df_feature and reflect in original df, new coordinates of moved feature are end_pt
    # Modify X and Y in df_feature and reflect in original df, new coordinates of moved feature are end_pt
    df.loc[df_feature.index, 'X'] = end_pt[0]
    df.loc[df_feature.index, 'Y'] = end_pt[1]
    df.loc[df_feature.index, 'Name'] = name
    
    feature_ID = df_feature['ID'].iloc[0]
    print("Feature being moved is ", feature_ID)

    # If the feature being moved is a PMT
    if str(feature_ID).endswith('00'):
        recalculate_all_bolts(df, feature_ID, end_pt[0], end_pt[1])

    # Bolt    
    else:
        recalculate_one_bolt(df, df_feature)

    return df_feature


def plot_labels(graph, df):
    labels = []

    # Loop over all points in the dataframe
    for index, row in df.iterrows():

        if row[1] != None:
            pmt_id = str(row[1])[:-3]
            bolt_id = str(row[1])[-2:]

            draw_x = row[2]
            draw_y = row[3]
            color = 'red' if bolt_id == '00' else 'yellow'
            text = pmt_id if bolt_id == '00' else bolt_id

            labels.append(graph.DrawText(text=text, location=(draw_x - 10, draw_y - 10), color=color))

    return labels


def erase_labels(graph, labels):
    for label in labels:
        graph.delete_figure(label)


def get_marker_center(graph, fig):
    current_coords = graph.get_bounding_box(fig)
    curr_x = (current_coords[0][0] + current_coords[1][0]) / 2
    curr_y = (current_coords[0][1] + current_coords[1][1]) / 2
    return curr_x, curr_y


def calc_row_col(df, new_ref):

    # calculate the vector from the reference PMT, new_ref to every other PMT in the dataframe (ends with '00')
    df_pmts = df[df['ID'].apply(lambda id: (str(id)[-2:] == '00'))]

    # get the x and y coordinates of the reference PMT
    x_ref = df[df['ID'] == new_ref]['X'].iloc[0]
    y_ref = df[df['ID'] == new_ref]['Y'].iloc[0]

    x_bound_left = x_ref - var.row_col_bound
    x_bound_right = x_ref + var.row_col_bound

    y_bound_bottom = y_ref - var.row_col_bound
    y_bound_top = y_ref + var.row_col_bound

    # make a list of all the PMTs with x-coordinates between x_bound_left and x_bound_right
    column = df_pmts[df_pmts['X'].apply(lambda x: x_bound_left < x < x_bound_right)]['ID'].tolist()

    # make a list of all the PMTs with y-coordinates between y_bound_bottom and y_bound_top
    row = df_pmts[df_pmts['Y'].apply(lambda y: y_bound_bottom < y < y_bound_top)]['ID'].tolist()

    ## organize the PMTs in the same row such in order of increasing x-coordinate
    row.sort(key=lambda id: df[df['ID'] == id]['X'].iloc[0])
    print("The number of PMTs in the same row as the reference PMT, new_ref, is", len(row))

    ## organize the PMTs in the same column such in order of increasing y-coordinate
    column.sort(key=lambda id: df[df['ID'] == id]['Y'].iloc[0])
    print("The number of PMTs in the same column as the reference PMT, new_ref, is", len(column))

    # put all the PMTs from row with x-coordinate less than x_ref in a list
    lesser_x_row = [id for id in row if df_pmts[df_pmts['ID'] == id]['X'].iloc[0] < x_ref]
    lesser_x_row.reverse()

    # put all the PMTs from row with x-coordinate greater than x_ref in a list
    greater_x_row = [id for id in row if df_pmts[df_pmts['ID'] == id]['X'].iloc[0] > x_ref]

    # put all the PMTs from column with y-coordinate less than y_ref in a list
    lesser_y_col = [id for id in column if df_pmts[df_pmts['ID'] == id]['Y'].iloc[0] < y_ref]

    # reverse the order of the PMTs in the column with y-coordinate less than y_ref
    lesser_y_col.reverse()

    # put all the PMTs from column with y-coordinate greater than y_ref in a list
    greater_y_col = [id for id in column if df_pmts[df_pmts['ID'] == id]['Y'].iloc[0] > y_ref]

    row = lesser_x_row + greater_x_row
    print("Final row is", row, ".")
    column = lesser_y_col + greater_y_col
    print("Final column is", column, ".")

    return df_pmts, lesser_x_row, greater_x_row, lesser_y_col, greater_y_col, row, column

def assign_adjacent_labels(df, pmt, label):
    # find pmt in the 'ID' column of the dataframe and check if an entry has been made in the 'Labels' column
    # if there is an entry, do nothing. If not, assign 'label' to it

    if df[df['ID'] == pmt]['Labels'].iloc[0] != None:
        return df
    else:
        df.at[df[df['ID'] == pmt].index[0], 'Labels'] = label
        print("Assigned label ", label, " to PMT ", pmt)

        # copy all entries in the 'ID' column that begin with the same 5 digits as pmt and don't end with '00' into the 'Labels' column
        # after copying, replace the first 5 digits with the string, label
        df.loc[df['ID'].apply(lambda id: (str(id)[:-3] == str(pmt)[:-3]) & (str(id)[-2] != '00')), 'Labels'] = str(label) + df['ID'].str[-3:]
        print("Assigned label ", label, " to bolts for this PMT ", pmt)

        return df

def autolabel(df, new_ref, label):

    # copy all entries in the 'ID' column that begin with the same 5 digits as new_ref and don't end with '00' into the 'Labels' column
    # after copying, replace the first 5 digits with the string, label
    df.loc[df['ID'].apply(lambda id: (str(id)[:-3] == str(new_ref)[:-3]) & (str(id)[-2:] != '00')), 'Labels'] = str(label) + df['ID'].str[-3:]

    # finding PMTs in the same row and column as reference PMT, new_ref
    df_pmts, lesser_x_row, greater_x_row, lesser_y_col, greater_y_col, row, column = calc_row_col(df, new_ref)

    row_label = int(label) # used as a buffer row label to assign labels to the PMTs in the same row as the reference PMT
    col_label = int(label) # used as a buffer column label to assign labels to the PMTs in the same column as the reference PMT

    # assign labels to the PMTs in the same row as the reference PMT, increasing by 51 each element
    for i in greater_x_row:
        row_label -= 51

        if df[df['ID'] == i]['Labels'].iloc[0] != None:
            continue

        else:
            df.at[df[df['ID'] == i].index[0], 'Labels'] = row_label

            # copy all entries in the 'ID' column that begin with the same 5 digits as i and don't end with '00' into the 'Labels' column
            # after copying, replace the first 5 digits with the string, row_label
            df.loc[df['ID'].apply(lambda id: (str(id)[:-3] == str(i)[:-3]) & (str(id)[-2:] != '00')), 'Labels'] = str(row_label) + df['ID'].str[-3:]
            print("Assigned label ", row_label, " to PMT ", i)

    row_label = int(label) # reset row_label to the original label

    # assign labels to the PMTs in lesser_x_row, decreasing by 51 each element
    for i in lesser_x_row:
        row_label += 51

        if df[df['ID'] == i]['Labels'].iloc[0] != None:
            continue

        else:
            df.at[df[df['ID'] == i].index[0], 'Labels'] = row_label

            # copy all entries in the 'ID' column that begin with the same 5 digits as i and don't end with '00' into the 'Labels' column
            # after copying, replace the first 5 digits with the string, row_label
            df.loc[df['ID'].apply(lambda id: (str(id)[:-3] == str(i)[:-3]) & (str(id)[-2:] != '00')), 'Labels'] = str(row_label) + df['ID'].str[-3:]
            print("Assigned label ", row_label, " to PMT ", i)

    # assign labels to the PMTs in the same column as the reference PMT, increasing by 1 each element
    for i in lesser_y_col:
        col_label -= 1

        if df[df['ID'] == i]['Labels'].iloc[0] != None:
            continue

        else:
            df.at[df[df['ID'] == i].index[0], 'Labels'] = col_label

            # copy all entries in the 'ID' column that begin with the same 5 digits as i and don't end with '00' into the 'Labels' column
            # after copying, replace the first 5 digits with the string, col_label
            df.loc[df['ID'].apply(lambda id: (str(id)[:-3] == str(i)[:-3]) & (str(id)[-2:] != '00')), 'Labels'] = str(row_label) + df['ID'].str[-3:]
            print("Assigned label ", col_label, " to PMT ", i)

    col_label = int(label) # reset col_label to the original label

    # assign labels to the PMTs in the same column as the reference PMT, decreasing by 1 each element
    for i in greater_y_col:
        col_label += 1

        if df[df['ID'] == i]['Labels'].iloc[0] != None:
            continue
        else:
            df.at[df[df['ID'] == i].index[0], 'Labels'] = col_label

            # copy all entries in the 'ID' column that begin with the same 5 digits as i and don't end with '00' into the 'Labels' column
            # after copying, replace the first 5 digits with the string, col_label
            df.loc[df['ID'].apply(lambda id: (str(id)[:-3] == str(i)[:-3]) & (str(id)[-2:] != '00')), 'Labels'] = str(row_label) + df['ID'].str[-3:]
            print("Assigned label ", col_label, " to PMT ", i)

    row = lesser_x_row + greater_x_row
    column = lesser_y_col + greater_y_col
    print("Sorted new row", lesser_x_row, greater_x_row)
    print("Sorted new column", lesser_y_col, greater_y_col)

    return df, new_ref, lesser_x_row, greater_x_row, lesser_y_col, greater_y_col, row, column

# def finish_labels(df, ref):
#
#     # calculate the row and column of the reference PMT
#     df_pmts, lesser_x_row, greater_x_row, lesser_y_col, greater_y_col, row, column = calc_row_col(df, ref)
#
#     # assign label to the reference PMT according to the labels of the PMTs in the same row and column
#     # get the labels of the PMTs in the same row and column as the reference PMT
#     less_x_labels = np.array(df['Labels'].iloc[lesser_x_row])
#     great_x_labels = np.array(df['Labels'].iloc[greater_x_row])
#     less_y_labels = np.array(df['Labels'].iloc[lesser_y_col])
#     great_y_labels = np.array(df['Labels'].iloc[greater_y_col])

    # get the first element of the


def autolabel_plot(df):

    # indicies of all the PMTs in the dataframe
    pmt_indicies = df[df['ID'].str.endswith('00')].index

    # get the x and y coordinates of the PMTs
    x = np.array(df['X'].iloc[pmt_indicies])
    y = np.array(df['Y'].iloc[pmt_indicies])

    # get the labels of the PMTs
    pmt_labels = np.array(df['Labels'].iloc[pmt_indicies])
    print("PMT Labels", pmt_labels)

    diag_plt.make_window(x, y, pmt_labels)

def pad_zeros(not_none_labels, df):

    # find the indicies where 'R' and 'theta' are nan and where they are not
    PMT_indicies = df[df['R'].isnull()].index # PMTs
    bolt_indicies = df[df['R'].notnull()].index # bolts

    # keep only matching indicies between PMT_indicies and not_none_labels in PMT_indicies
    # repeat with bolt_indicies
    PMT_indicies = [i for i in PMT_indicies if i in not_none_labels]
    bolt_indicies = [i for i in bolt_indicies if i in not_none_labels]

    # convert the values at the indicies in the 'ID' column to strings and pad them with zeros
    # concatenate '00' to the end of the strings
    df.loc[PMT_indicies, 'ID'] = df.loc[PMT_indicies, 'ID'].astype(str).str.pad(width=5, side='left', fillchar='0') + '-00' # pmts

    # fix bolt labels with padding and correct number
    df.loc[bolt_indicies, 'ID'] = df.loc[bolt_indicies, 'ID'].str[:-3].str.pad(width=5, side='left', fillchar='0') + '-' + df.loc[bolt_indicies, 'ID'].str[-2:] # bolts

    return df


def get_next_ref(df, count, row_column):
    # look for the next PMT in the row/column
    index = df[df['ID'] == row_column[count]].index[0]

    # get the ID of the PMT at index and label
    new_ref, new_label = df['ID'].iloc[index], df['Labels'].iloc[index]
    print("New reference PMT: ", new_ref, "and label: ", new_label)

    return new_ref, new_label

def finalize_df(df):

    # get the indices for the entries which don't have None in the 'Labels' column
    not_none_indicies = df[df['Labels'].notnull()].index

    df.loc[df['Labels'].notnull(), 'ID'] = df.loc[df['Labels'].notnull(), 'Labels']

    df = df.drop(columns=['Labels'])

    df = pad_zeros(not_none_indicies, df)

    print("Final dataframe correct ID format. ", df.to_string())
    return df