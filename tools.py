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


def InteractiveSlicer3D(img3D, axis, ax, coor_system):
    """
    Rozšířená funkce umožňující interaktivní výběr bodů a jejich propojení přímkou,
    s možností mazání pravým tlačítkem myši. Kliknuté body jsou ukládány s jejich (x, y, z) souřadnicemi.
    """
    if coor_system == 'Original':
        g_name = ['Axiální rovina', 'Sagitální rovina', 'Koronální rovina']
    elif coor_system == 'SA':
        g_name = ['pseudoVLA', 'SA', 'PseudoHLA']

    if axis == 'axial':
        current_frame = img3D.shape[2] // 2
        num_frames = img3D.shape[2]
        slice_func = lambda i: img3D[:, :, i]
        plain = g_name[0]
    elif axis == 'sagital':
        current_frame = img3D.shape[0] // 2
        num_frames = img3D.shape[0]
        slice_func = lambda i: img3D[i, :, :]
        plain = g_name[1]
    elif axis == 'coronal':
        current_frame = img3D.shape[1] // 2
        num_frames = img3D.shape[1]
        slice_func = lambda i: img3D[:, i, :]
        plain = g_name[2]

    frame_img = ax.imshow(slice_func(current_frame), cmap='gray')
    ax.set_title(f'{plain} - snímek: {current_frame + 1}/{num_frames}')
    ax.axis('off')

    points = []  # Seznam pro ukládání vybraných bodů s jejich souřadnicemi
    markers = []  # Seznam pro uložené markery
    line = None  # Uchování jediné čáry

    def on_scroll(event):
        nonlocal current_frame
        if event.inaxes == ax:
            if event.step > 0:
                current_frame = (current_frame + 1) % num_frames
            elif event.step < 0:
                current_frame = (current_frame - 1) % num_frames

            frame_img.set_data(slice_func(current_frame))
            ax.set_title(f'{plain} - snímek: {current_frame + 1}/{num_frames}')
            ax.figure.canvas.draw_idle()

    def on_click(event):
        nonlocal points, markers, line, current_frame
        if event.inaxes == ax and event.button == MouseButton.LEFT:
            if len(points) < 2:
                x, y = event.xdata, event.ydata
                if axis == 'axial':
                    z = current_frame
                elif axis == 'sagital':
                    y, z = event.ydata, current_frame
                elif axis == 'coronal':
                    x, z = event.xdata, current_frame

                points.append((x, y, z))
                marker, = ax.plot(x, y, 'ro')
                markers.append(marker)

                if len(points) == 2:
                    if line:
                        line.remove()
                    x_vals, y_vals = zip(*[(p[0], p[1]) for p in points])
                    line, = ax.plot(x_vals, y_vals, 'r-')  # Vykreslení přímky

                ax.figure.canvas.draw_idle()

        elif event.inaxes == ax and event.button == MouseButton.RIGHT:
            points.clear()
            for marker in markers:
                marker.remove()
            markers.clear()
            if line:
                line.remove()
                line = None
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)
    ax.figure.canvas.mpl_connect('button_press_event', on_click)

    return points


### Pro načtení nových dat:
# ssd_path = r'D:\Original'
# json_path = os.path.join(ssd_path, 'transformInfo.json')
#
# with open(json_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# A_transform = np.array(data[0]['TransformSurvToSA']['A'])[:3, :3]
# Folder_ID = data[0]['FolderID']
# nib_path = os.path.join(ssd_path,
#                        Folder_ID,
#                        'Scout',
#                        's3D_BTFE_NAV.nii')
#
# nii_data = nib.load(nib_path)
# nii_img = LoadniiAndPad(nii_data, (0, 0, 400))
# nii_img = RotationAffine3D(nii_img, A_transform)
#
# x, y, z = nii_img.shape
# n = 150
# nii_img = nii_img[x//2 - n: x//2 + n,
#                   y//2 - n: y//2 + n,
#                   z//2 - n: z//2 + n]
#
# nii_object = nib.Nifti1Image(nii_img, np.eye(4))
# nib.save(nii_object, 'tools.nii')
###

nii_object = nib.load('tools.nii')
nii_img = nii_object.get_fdata()

fig, ax = plt.subplots(1, 3)
Slicer3D(nii_img, 'axial', ax[0], 'SA')
points = InteractiveSlicer3D(nii_img, 'sagital', ax[1], 'SA')
Slicer3D(nii_img, 'coronal', ax[2], 'SA')
plt.show()

points = np.array(points)
points = np.round(points, 0)
print(points)