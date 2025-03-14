import numpy as np
import os
import json
import nibabel as nib
import matplotlib.pyplot as plt

from utils import Slicer3D, LoadniiAndPad, RotationAffine3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


class InteractiveSlicer3D:
    def __init__(self, img3D, axis, ax, coor_system):
        """
        Interaktivní prohlížeč 3D obrazu s možností výběru bodů a posouvání mezi řezy.
        Kliknutím levým tlačítkem myši se označí bod a uloží do seznamu.
        Kliknutím pravým tlačítkem se body vymažou.
        """
        self.img3D = img3D
        self.axis = axis
        self.ax = ax
        self.points = []  # Ukládané body
        self.markers = []  # Seznam markerů
        self.current_frame = 0  # Počáteční řez

        # Nastavení pojmenování rovin
        if coor_system == 'Original':
            g_name = ['Axiální rovina', 'Sagitální rovina', 'Koronální rovina']
        elif coor_system == 'SA':
            g_name = ['pseudoVLA', 'SA', 'PseudoHLA']
        else:
            g_name = ['Axial', 'Sagittal', 'Coronal']

        # Nastavení řezové osy
        if axis == 'axial':
            self.current_frame = img3D.shape[2] // 2
            self.num_frames = img3D.shape[2]
            self.slice_func = lambda i: img3D[:, :, i]
            self.plane_name = g_name[0]
        elif axis == 'sagittal':
            self.current_frame = img3D.shape[0] // 2
            self.num_frames = img3D.shape[0]
            self.slice_func = lambda i: img3D[i, :, :]
            self.plane_name = g_name[1]
        elif axis == 'coronal':
            self.current_frame = img3D.shape[1] // 2
            self.num_frames = img3D.shape[1]
            self.slice_func = lambda i: img3D[:, i, :]
            self.plane_name = g_name[2]
        else:
            raise ValueError("Invalid axis. Choose from 'axial', 'sagittal', 'coronal'.")

        # Zobrazení prvního řezu
        self.frame_img = self.ax.imshow(self.slice_func(self.current_frame), cmap='gray')
        self.ax.set_title(f'{self.plane_name} - snímek: {self.current_frame + 1}/{self.num_frames}')
        self.ax.axis('off')

        # Připojení event handlerů
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        """ Funkce pro posouvání mezi řezy pomocí kolečka myši """
        if event.inaxes == self.ax:
            if event.step > 0:  # Posun vpřed
                self.current_frame = (self.current_frame + 1) % self.num_frames
            elif event.step < 0:  # Posun zpět
                self.current_frame = (self.current_frame - 1) % self.num_frames

            # Aktualizace obrazu
            self.frame_img.set_data(self.slice_func(self.current_frame))
            self.ax.set_title(f'{self.plane_name} - snímek: {self.current_frame + 1}/{self.num_frames}')
            self.ax.figure.canvas.draw_idle()

    def on_click(self, event):
        """ Funkce pro interaktivní označování bodů a mazání """
        if event.inaxes == self.ax:
            if event.button == MouseButton.LEFT:  # Levé tlačítko - přidání bodu
                x, y = event.xdata, event.ydata
                if self.axis == 'axial':
                    z = self.current_frame
                elif self.axis == 'sagittal':
                    y, z = event.ydata, self.current_frame
                elif self.axis == 'coronal':
                    x, z = event.xdata, self.current_frame

                self.points.append((x, y, z))  # Uložení bodu
                marker, = self.ax.plot(x, y, 'ro')  # Označení bodu červeným kroužkem
                self.markers.append(marker)
                self.ax.figure.canvas.draw_idle()

            elif event.button == MouseButton.RIGHT:  # Pravé tlačítko - mazání bodů
                self.points.clear()
                for marker in self.markers:
                    marker.remove()
                self.markers.clear()
                self.ax.figure.canvas.draw_idle()

    def get_points(self):
        """ Funkce pro získání všech uložených bodů """
        return self.points

