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


    # Major ticks every 500, minor ticks every 100
    major_ticks_x = np.arange(0, 4000, 500)
    minor_ticks_x = np.arange(0, 4000, 100)

    major_ticks_y = np.arange(0, 2750, 500)
    minor_ticks_y = np.arange(0, 2750, 100)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # Add corresponding grid
    ax.grid(which='both', axis='both', linestyle='--')

    # automatically adjust how much of the plot is shown to fit all points with a small margin
    ax.set_xlim([min(x) - 100, max(x) + 100])
    ax.set_ylim([min(y) - 100, max(y) + 100])

    # set axis labels
    ax.set_xlabel('x [pixels]')
    ax.set_ylabel('y [pixels]')

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
