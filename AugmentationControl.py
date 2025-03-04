import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from utils import Slicer3D, LoadniiAndPad

save_dir_data = r'D:\DataSet\Data'
save_dir_gt = r'D:\DataSet\GT'

selection = 0

match selection:
    case 0:
        ### KONTROLA 0 = kontrola orignál a gt získané rotací
        nii_da = nib.load(os.path.join(save_dir_data, '000000', 'img3D.nii'))
        nii_da_img = LoadniiAndPad(nii_da)
        print(nii_da_img.shape)

        nii_gt = nib.load(os.path.join(save_dir_gt, '000000', 'img3D.nii'))
        nii_gt_img = LoadniiAndPad(nii_gt)

        fig, ax = plt.subplots(2, 3)
        # AUGMENTOVANÝ OBRAZ - SA:
        Slicer3D(nii_da_img, axis='axial', ax=ax[0][0], coor_system='Original')
        Slicer3D(nii_da_img, axis='coronal', ax=ax[0][1], coor_system='Original')
        Slicer3D(nii_da_img, axis='sagital', ax=ax[0][2], coor_system='Original')
        # ORIGINÁLNÍ OBRAZ - SA:
        Slicer3D(nii_gt_img, axis='axial', ax=ax[1][0], coor_system='SA')
        Slicer3D(nii_gt_img, axis='coronal', ax=ax[1][1], coor_system='SA')
        Slicer3D(nii_gt_img, axis='sagital', ax=ax[1][2], coor_system='SA')
        plt.show()

    case 1:
        nii_gt1 = nib.load(os.path.join(save_dir_gt, '003006', 'img3D.nii'))
        nii_gt_img1 = LoadniiAndPad(nii_gt1)

        print(nii_gt_img1.shape)

        nii_gt2 = nib.load(os.path.join(save_dir_gt, '000006', 'img3D.nii'))
        nii_gt_img2 = LoadniiAndPad(nii_gt2)

        fig, ax = plt.subplots(2, 3)
        # AUGMENTOVANÝ OBRAZ - SA:
        Slicer3D(nii_gt_img1, axis='axial', ax=ax[0][0], coor_system='SA')
        Slicer3D(nii_gt_img1, axis='coronal', ax=ax[0][1], coor_system='SA')
        Slicer3D(nii_gt_img1, axis='sagital', ax=ax[0][2], coor_system='SA')
        # ORIGINÁLNÍ OBRAZ - SA:
        Slicer3D(nii_gt_img2, axis='axial', ax=ax[1][0], coor_system='SA')
        Slicer3D(nii_gt_img2, axis='coronal', ax=ax[1][1], coor_system='SA')
        Slicer3D(nii_gt_img2, axis='sagital', ax=ax[1][2], coor_system='SA')
        plt.show()