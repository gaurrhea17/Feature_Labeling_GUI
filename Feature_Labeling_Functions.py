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
# matplotlib.use("TkAgg") ##importing matplotlib with Tkinter rather than another framework like Qt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import Feature_Labeling_Variables as var

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

def undistort_points(df, newcameramtx):
    # Unpack camera parameters
    k1, k2, p1, p2, k3 = var.k1, var.k2, var.p1, var.p2, var.k3  # #k1, k2 and k3 are radial distortion coefficients,
    # p1 and p2 are tangential distortion coefficients

    fx, fy, cx, cy = var.fx, var.fy, var.cx, var.cy

    # Convert distorted points to homogeneous coordinates
    distorted_points = df[['X', 'Y']].values.tolist()
    distorted_points = np.array(distorted_points)

    dst = cv.undistortPoints(distorted_points, newcameramtx, var.dist)

    return dst


def distort_points(undistorted_points, distortion_coeffs, camera_params):
    # Convert undistorted points to homogeneous coordinates
    undistorted_points_homogeneous = np.vstack((undistorted_points, np.ones((1, len(undistorted_points)))))

    # Perform projection using camera matrix
    distorted_points_homogeneous = np.dot(camera_params, undistorted_points_homogeneous)

    # Convert distorted points to non-homogeneous coordinates
    distorted_points = distorted_points_homogeneous[:2] / distorted_points_homogeneous[2]

    # Apply radial and tangential distortion
    r_square = distorted_points[0] ** 2 + distorted_points[1] ** 2
    radial_distortion = 1 + distortion_coeffs[0] * r_square + distortion_coeffs[1] * r_square ** 2
    tangential_distortion_x = 2 * distortion_coeffs[2] * distorted_points[0] * distorted_points[1] + distortion_coeffs[
        3] * (r_square + 2 * distorted_points[0] ** 2)
    tangential_distortion_y = distortion_coeffs[2] * (r_square + 2 * distorted_points[1] ** 2) + 2 * distortion_coeffs[
        3] * distorted_points[0] * distorted_points[1]

    distorted_points = distorted_points * radial_distortion + np.array(
        [tangential_distortion_x, tangential_distortion_y])

    return distorted_points


def undistort_img(img_path, mtx, dist):  ## uses bilinear interpolation
    '''This function is used to undistort the image to display in the GUI.'''

    img = cv.imread(img_path)
    h, w = img.shape[:2]

    '''Free scaling parameter. If it is -1 or absent, the function performs the default scaling. 
    Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are 
    zoomed and shifted so that only valid pixels are visible (no black areas after rectification). 
    alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the 
    original images from the cameras are retained in the rectified images (no source image pixels are lost). 
    Any intermediate value yields an intermediate result between those two extreme cases.'''

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), -1, (w, h))  ## alpha = 1

    ## undistorting
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)  # can include argument, newcameramtx to this function

    ## can crop the image (eliminates black areas after rectification, especially when alpha=1)
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w] ## array of uint8

    # Convert the undistorted image to the correct type bytes
    undistorted_img = cv.imencode('.png', dst)[1].tobytes()

    return undistorted_img, newcameramtx


## User selected an image to display
def disp_image(window, values, fnames, location, undistort):
    """ This function is called when the user selects an image they wish to display.
    This can be either when an image is selected from the listbox or the "Next" 
    or "Previous" buttons are selected. 
    
    The graph element is then updated with the image and the filename is also updated. """

    if location == 0:
        fname = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )

        index = np.where(np.array(fnames) == values["-FILE LIST-"][0])[0][0]

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

    if undistort:
        ## Undistorting the image first
        focal_length, principle_point, radial_distortion, tangential_distortion, mtx, dist = camera_model(var.fx,
                                                                                                          var.fy,
                                                                                                          var.cx,
                                                                                                          var.cy,
                                                                                                          var.k1,
                                                                                                          var.k2,
                                                                                                          var.p1,
                                                                                                          var.p2,
                                                                                                          var.skew)
        data, newcameramtx = undistort_img(fname, mtx, dist)

    elif not undistort:
        ## Converting image to array to data -- for display of distorted images
        data = array_to_data(im, fname)
        newcameramtx = None

    ids = window["-GRAPH-"].draw_image(data=data, location=(0, var.height))

    with open(fname, 'rb') as f:
        im_bytes = f.read()
    pil_image = Image.open(io.BytesIO(im_bytes))

    return im, pil_image, fname, ids, newcameramtx


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


def draw_pts(graph, df, undistort, scale=1):
    for index, row in df.iterrows():

        if undistort:
            draw_coords = (float(row[4]) * scale, float(row[5]) * scale)
        elif not undistort:
            draw_coords = (float(row[2]) * scale, float(row[3]) * scale)
        color = 'yellow'
        size = 6
        if str(row[1]).endswith('00'):  # PMT
            color = 'red'
            size = 8

        graph.draw_point(draw_coords, color=color, size=size)


def autoload_pts(fname, name, mtx, undistort):
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

    ## Undistort the points and add to dataframe
    # u_coords = undistort_points(df, mtx)
    # print(u_coords)
    # print("Undistortion worked. Shape of u_coords is ", u_coords.shape)
    #
    # # insert x coordinates from u_coords into dataframe at column 4
    # # insert y coordinates from u_coords into dataframe at column 5
    # df.insert(4, 'Undistort_X', u_coords[:, 0])
    # df.insert(5, 'Undistort_Y', u_coords[:, 1])


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

            if undistort:
                df.at[index, 'R'] = np.sqrt((float(row[4]) - pmt_x) ** 2 + (float(row[5]) - pmt_y) ** 2)
                df.at[index, 'theta'] = angle_to((float(row[4]), float(row[5])), (pmt_x, pmt_y))
            elif not undistort:
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

def modify_label(df, position, id_new):
    df_feature = get_current_feature(df, position)

    # Directly modify ID of feature in original dataframe
    df.loc[df_feature.index, 'ID'] = id_new

    # Update bolt_r and theta for this feature if it is a bolt
    if not str(id_new).endswith('00'):
        recalculate_one_bolt(df, df_feature)

    # Warning: no treatment of existing bolts if modified label is a PMT
    # But probably want to use a function that changes all the previous bolt labels, if any
    # (i.e from the new PMT auto-labeling function)

    return df_feature

def del_point(df, position):
    df_feature = get_current_feature(df, position)

    df.drop(df_feature.index, inplace=True)

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
    # Check for duplicate IDs
    if len(df['ID'].unique()) != len(df.index):

        # List duplicate IDs
        for i, dup_id in reverseEnum(df['ID'].unique()):
            df_unique = df[df['ID'] == dup_id]
            if len(df_unique.index) > 1:
                x = df_unique['X'].iloc[0]
                y = df_unique['Y'].iloc[0]

                df.drop_duplicates(subset=df.columns[1], keep="first",
                                   inplace=True)  ## removing duplicate based on ID and keeping the first entry
                df.reset_index(drop=True, inplace=True)  ## resetting indices after removing duplicates

                sg.popup("Duplicate ID: ", dup_id, " at (", str(x), ", " + str(y) + ") will be removed.")
    return df


def write_coords_to_file(df, filename):
    df = duplicate_check(df)
    df = df.sort_values(by=['ID'])

    # Write X/Y column with precision 
    df['X'] = df['X'].map(lambda x: '{:.1f}'.format(x))
    df['Y'] = df['Y'].map(lambda y: '{:.1f}'.format(y))

    headers = ["Img", "ID", "X", "Y", "Name"]
    df[headers].to_csv(filename, index=False, header=False, sep='\t')
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

    df_bolts = df[df['ID'].apply(lambda id: (id[:5] == New_ID[:5]) & (float(id[-2:]) > 0))]

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

    # Modify X and Y in df_feature and reflect in original df
    df.loc[df_feature.index, 'X'] = end_pt[0]
    df.loc[df_feature.index, 'Y'] = end_pt[1]
    df.loc[df_feature.index, 'Name'] = name
    
    feature_ID = df_feature['ID'].iloc[0]
    # print("Feature being moved is ", feature_ID)

    # If the feature being moved is a PMT
    if str(feature_ID).endswith('00'):
        recalculate_all_bolts(df, feature_ID, end_pt[0], end_pt[1])

    # Bolt    
    else:
        recalculate_one_bolt(df, df_feature)

    return df_feature


def plot_labels(graph, df, undistort):
    labels = []

    # Loop over all points in the dataframe
    for index, row in df.iterrows():

        pmt_id = str(row[1])[:5]
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

def autolabel(df, new_ref, label):

    # calculate the vector from the reference PMT, new_ref to every other PMT in the dataframe (ends with '00')
    df_pmts = df[df['ID'].apply(lambda id: (id[-2:] == '00'))]

    df_pmts['vector'] = df_pmts.apply(lambda row: [row['X'] - df_pmts[df_pmts['ID'] == new_ref]['X'].iloc[0],
                                                    row['Y'] - df_pmts[df_pmts['ID'] == new_ref]['Y'].iloc[0]], axis=1)


    # calculate the angle between the vector and the x-axis
    # If the angle is less than 45 degrees, then the PMT is in the same row as the reference PMT, new_ref.
    df_pmts['angle'] = df_pmts.apply(lambda row: np.arctan(row['vector'][1] / row['vector'][0]) * 180 / np.pi, axis=1)

    # make a list of all the PMTs in the same row as the reference PMT
    row = df_pmts[df_pmts['angle'].apply(lambda angle: abs(angle) < 7)]['ID'].tolist()

    # calculate the angle between the vector and the y-axis
    # If the angle is less than 45 degrees, then the PMT is in the same column as the reference PMT, new_ref.
    df_pmts['angle'] = df_pmts.apply(lambda row: np.arctan(row['vector'][0] / row['vector'][1]) * 180 / np.pi, axis=1)

    # make a list of all the PMTs in the same column as the reference PMT
    column = df_pmts[df_pmts['angle'].apply(lambda angle: abs(angle) < 7)]['ID'].tolist()

    ## organize the PMTs in the same row such in order of increasing x-coordinate
    row.sort(key=lambda id: df[df['ID'] == id]['X'].iloc[0])

    ## organize the PMTs in the same column such in order of decreasing y-coordinate
    column.sort(key=lambda id: df[df['ID'] == id]['Y'].iloc[0], reverse=True)

    row_label = int(label) # used as a buffer row label to assign labels to the PMTs in the same row as the reference PMT
    col_label = int(label) # used as a buffer column label to assign labels to the PMTs in the same column as the reference PMT

    # make new list of PMTs in the same row as the reference PMT with x coordinate greater than the reference PMT
    new_row = []
    for i in row:
        if df[df['ID'] == i]['X'].iloc[0] > df[df['ID'] == new_ref]['X'].iloc[0]:
            new_row.append(i)

    # make new list of PMTs in the same column as the reference PMT with y coordinate less than the reference PMT
    new_column = []
    for i in column:
        if df[df['ID'] == i]['Y'].iloc[0] < df[df['ID'] == new_ref]['Y'].iloc[0]:
            new_column.append(i)

    # assign labels to the PMTs in the same row as the reference PMT, increasing by 51 each element
    for i in new_row:
        row_label += 51
        # if no label, assign a label
        if df[df['ID'] == i]['Labels'].iloc[0] == row_label:
            continue
        else:
            df.at[df[df['ID'] == i].index[0], 'Labels'] = row_label
            print("Assigned label ", row_label, " to PMT ", i)


    # assign labels to the PMTs in the same column as the reference PMT, increasing by 1 each element
    for i in new_column:
        col_label += 1
        # if no label exists for this PMT, assign a label
        if df[df['ID'] == i]['Labels'].iloc[0] == col_label:
            continue
        else:
            df.at[df[df['ID'] == i].index[0], 'Labels'] = col_label
            print("Assigned label ", col_label, " to PMT ", i)

    print("Sorted new row", new_row)
    print("Sorted new column", new_column)

    # print("New dataframe with labels", df.to_string())

    return df, new_ref, row, column
