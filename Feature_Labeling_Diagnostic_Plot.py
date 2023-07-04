import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib

matplotlib.use('TkAgg')


def figure(x, y, labels):
    fig = matplotlib.figure.Figure(dpi=100)
    ax = fig.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(labels):
        ax.annotate(labels[i], (x[i], y[i]))

    return fig, ax


def draw_figure(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return tkcanvas


def make_window(x, y, labels):
    layout = [[sg.Text('Diagnostic Plot')],
              [sg.Canvas(key='-CANVAS-')],
              [sg.Button('Ok')]]
    window = sg.Window('Diagnostics Plot', layout, finalize=True, resizable=True, element_justification='center',
                       font='Helvetica 18')

    # add the plot to the window
    fig, ax = figure(x, y, labels)
    tkcanvas = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    event, values = window.read()
    window.close()
