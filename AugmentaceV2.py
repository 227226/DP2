# import balíčků:
import os
import numpy as np
import json
import nibabel as nib
import scipy.ndimage as ndi

from scipy.ndimage import center_of_mass
from numpy.linalg import inv
from random import sample

# import z funkcí
from utils import RotationAffine3D,\
                  AffineMatrix,\
                  LoadniiAndPad,\
                  GaussianNoise3D,\
                  RestrictImg,\
                  ScaleOriginalImageShape,\
                  AnglesVariations,\
                  CreateCube,\
                  CreateBoundingBox,\
                  CubeRotation,\
                  PermittedState

# nastavení hl. složky, ze které budou načítána data:
root_dir = r'D:\Original'
# načtené informací o obrazech:
patient_info_path = os.path.join(root_dir, 'transformInfo.json')

# nastavení složek pro ukládání dat:
# složka pro data:
save_dir_data = r'D:\DataSetV2\Data'
# složka pro ground truth:
save_dir_gt = r'D:\DataSetV2\GT'

# elementární složky pro ukládání obrazů:
os.makedirs(save_dir_data, exist_ok=True)
os.makedirs(save_dir_gt, exist_ok=True)

# načtené slovníku s informacemi:
with open(patient_info_path, 'r', encoding='utf-8') as data:
    transform_info = json.load(data)

# definice faktoru augmenace (udává počet kopií originálu, jež budou vytvořeny):
augmentation_factor = 10
# pomocné proměnné pro sledování procesu:
whole_number_of_images = len(transform_info) * augmentation_factor + len(transform_info)
k = 1

# inicializace pro ukládání informací o nových obrazech:
augmented_dataset_and_information = []

# iterace přes originály:
for i in range(len(transform_info)):
    # načtení transformační matice (originál):
    transform_A = np.array(transform_info[i]['TransformSurvToSA']['A'])[:3, :3]
    transform_A_original = transform_info[i]['TransformSurvToSA']['A']
    # načtení škálovací matice (originál):
    scaling_A = np.array(transform_info[i]['TransformScout']['A'])[:3, :3]
    scaling_A_original = transform_info[i]['TransformScout']['A']
    # výpočet rotační matice (ortogonalita {A@inv(A)=I} a normalita {det(A)=1} byly ověřeny):
    rotate_A = scaling_A @ transform_A
    rotate_A_orig = np.eye(4)
    rotate_A_orig[:3, :3] = rotate_A

    # definice cesty k obrazu:
    nii_path = os.path.join(transform_info[i]['pathOrigScoutnii'])
    # načtení nifti dat:
    nii_data = nib.load(nii_path)
    # načtení obrazových dat ve formě matice (x, y, z) a padding, je-li potřeba:
    z_pad = 200
    nii_img = LoadniiAndPad(nii_data, (0, 0, z_pad))
    # - padding je proveden, aby vlivem škálování a následné rotace nedošlo ke ztrátě části obrazu
    # škálování obrazu pro další postup:
    scaled_nii_img = RotationAffine3D(nii_img, inv(scaling_A))
    # provedení ořezání obrazu na platnou velikost:
    new_coors = ScaleOriginalImageShape(nii_img, z_pad, scaling_A)

    scaled_nii_img_restricted = RestrictImg(nii_img,
                                            (new_coors[0] / nii_img.shape[0],
                                             new_coors[1] / nii_img.shape[1],
                                             new_coors[2] / nii_img.shape[2])
                                                                             )
    # inicializace segmentační masky:
    segmented_img = np.zeros(scaled_nii_img_restricted.shape, dtype=np.uint8)  # dtype=np.uint8 -> 0 a 1
    # segmentace prahováním:
    segmented_img[(scaled_nii_img_restricted > 80000) & (scaled_nii_img_restricted < 140000)] = 1  # 65000 až 120000
    # labelování:
    labeled_mask, num_features = ndi.label(segmented_img)
    # spočítá velikost segmentů:
    sizes = ndi.sum(segmented_img, labeled_mask, range(num_features + 1))
    # vezme největší segment kromě pozadí:
    max_label = np.argmax(sizes[1:]) + 1  # +1 protože ignorujeme pozadí
    # začištění masky
    cleaned_mask = (labeled_mask == max_label).astype(np.uint8)
    # výpočet centroidu (přibližný střed srdce):
    centroid = center_of_mass(cleaned_mask)
    # vytažení jednotlivých koordinátů:
    x, y, z = centroid
    # převod do formátů integer a přidání penalizace:
    x_c = int(x) + 15
    y_c = int(y) + 10
    z_c = int(z) + 10
    # definice středu:
    center = (x_c, y_c, z_c)
    # definice velikosti obrazu:
    shape = scaled_nii_img_restricted.shape

    # výpočet všech úhlových variací na základě povolených stavů, které jsou
    # definovány přímo uvnitř funkce AnglesVariations:
    angles_variations = AnglesVariations()
    # vytvořní pomocné krychle, která vymezuje prostor původního obrazu,
    # umožňuje tak ověřit, zda se bounding box nachází uvnitř tohoto prostoru:
    points, faces = CreateCube(scaled_nii_img_restricted, scaled_nii_img_restricted)
    # inicializace bounding boxu
    bb, bb_faces = CreateBoundingBox((190, 190, 170), center)

    # inicializace proměnné pro ukládání akceptovatelných úhlových rotací:
    acceptable_angles = []
    # iterace přes úhlové rotce, transformace pomocné krychle a ověření, zda se
    # při dané rotaci nachází bb uvnitř rotovaného prostoru pomocí daných úhlů:
    for angles in angles_variations:
        A = AffineMatrix(angles, 'xyz')
        points1, faces1 = CubeRotation(points, inv(A), center)
        permitted_angles = PermittedState(points1, bb, angles)
        acceptable_angles.append(permitted_angles)
    # odfiltrování všech původních rotací:
    filtered_angles = [arr for arr in acceptable_angles if not np.array_equal(arr, np.array([0, 0, 0]))]
    # náhodný výběr filtrovaných náhodných rotací, podle počtu augmentací:
    num_samples = min(augmentation_factor, len(filtered_angles))
    random_angles = sample(filtered_angles, num_samples)


    # zašumění obrazu gaussovským šumem:
    scaled_nii_img_restricted = GaussianNoise3D(scaled_nii_img_restricted, 0.01)

    # sekce pro ukládání originálu:
    AugmentID = "{:03}".format(0)
    OriginalID = "{:03}".format(i)
    FolderID = AugmentID + OriginalID
    info = {'ID': {'FolderID': FolderID,
                   'AugmentID': AugmentID,
                   'OriginalID': OriginalID},
            'A': {'Original_A': transform_A_original,
                  'Scaling_A': scaling_A_original,
                  'Rotation_A': rotate_A_orig.tolist(),
                  'Augment_A': np.eye(4).tolist(),
                  'Augmented_A': rotate_A_orig.tolist(),
                  'Angles': (0, 0, 0)
                  },
            'Paths': {'data': os.path.join(save_dir_data, FolderID, 'img3D.nii'),
                      'gt': os.path.join(save_dir_gt, FolderID, 'img3D.nii')
                      }
            }

    # přidání informací ve formě slovníku do seznamu, pro následné uložení do JSON:
    augmented_dataset_and_information.append(info)

    # složky pro ukládání konkrétních obrazů:
    os.makedirs(os.path.join(save_dir_data, FolderID), exist_ok=True)
    os.makedirs(os.path.join(save_dir_gt, FolderID), exist_ok=True)

    # ukládání generovaného obrazu do formátu nifti:
    nii = nib.Nifti1Image(scaled_nii_img_restricted, np.eye(4))
    nib.save(nii, os.path.join(save_dir_data, FolderID, 'img3D.nii'))

    # načtení a provedení rotace, generování ground truth pro porovnání, uložení:
    nii_gt = nib.load(os.path.join(save_dir_data, FolderID, 'img3D.nii'))
    img_gt = LoadniiAndPad(nii_gt, (100, 100, 100))
    img_gt_rot = RotationAffine3D(img_gt, rotate_A)
    img_gt_nii = nib.Nifti1Image(img_gt_rot, np.eye(4))
    nib.save(img_gt_nii, os.path.join(save_dir_gt, FolderID, 'img3D.nii'))

    print(f'Vytvořeno {k}/{whole_number_of_images} 3D obrazů.')
    k += 1

    for j in range(augmentation_factor):
        # výpočet augmentační matice pro definovanou rotační konvenci a definované úhly v jednotlivých osách:
        augment_A = AffineMatrix(random_angles[j], 'xyz')
        augment_A_orig = np.eye(4)
        augment_A_orig[:3, :3] = augment_A
        # výpočet augmentované matice = ground_truth pro model:
        augmented_A = augment_A @ rotate_A
        augmented_A_orig = np.eye(4)
        augmented_A_orig[:3, :3] = augmented_A
        # transformace obrazu na augmentovaný obraz:
        augmented_nii_img = RotationAffine3D(scaled_nii_img, inv(augment_A))
        # provedení ořezání obrazu na platnou velikost:
        augmented_nii_img_restricted = RestrictImg(augmented_nii_img, (0.32, 0.32, 0.54))
        # zašumění obrazu gaussovským šumem:
        augmented_nii_img_restricted = GaussianNoise3D(augmented_nii_img_restricted, 0.01)

        # sekce pro ukládání augmentací:
        AugmentID = "{:03}".format(j+1)
        OriginalID = "{:03}".format(i)
        FolderID = AugmentID + OriginalID
        info = {'ID': {'FolderID': FolderID,
                       'AugmentID': AugmentID,
                       'OriginalID': OriginalID},
                'A': {'Original_A': transform_A_original,
                      'Scaling_A': scaling_A_original,
                      'Rotation_A': rotate_A_orig.tolist(),
                      'Augment_A': augment_A_orig.tolist(),
                      'Augmented_A': augmented_A_orig.tolist(),
                      'Angles': random_angles[j]
                      },
                'Paths': {'data': os.path.join(save_dir_data, FolderID, 'img3D.nii'),
                          'gt': os.path.join(save_dir_gt, FolderID, 'img3D.nii')
                          }
                }

        # přidání informací ve formě slovníku do seznamu, pro následné uložení do JSON:
        augmented_dataset_and_information.append(info)

        # složky pro ukládání konkrétních obrazů:
        os.makedirs(os.path.join(save_dir_data, FolderID), exist_ok=True)
        os.makedirs(os.path.join(save_dir_gt, FolderID), exist_ok=True)

        # ukládání generovaného obrazu do formátu nifti:
        nii = nib.Nifti1Image(augmented_nii_img_restricted, np.eye(4))
        nib.save(nii, os.path.join(save_dir_data, FolderID, 'img3D.nii'))

        # načtení a provedení rotace, generování ground truth pro porovnání, uložení:
        nii_gt = nib.load(os.path.join(save_dir_data, FolderID, 'img3D.nii'))
        img_gt = LoadniiAndPad(nii_gt, (100, 100, 100))
        img_gt_rot = RotationAffine3D(img_gt, augmented_A)
        img_gt_nii = nib.Nifti1Image(img_gt_rot, np.eye(4))
        nib.save(img_gt_nii, os.path.join(save_dir_gt, FolderID, 'img3D.nii'))

        print(f'Vytvořeno {k}/{whole_number_of_images} 3D obrazů.')
        k += 1

# uložení json souboru s informace o provedené augmentaci:
with open(os.path.join(r'D:\DataSet', 'Info.json'), "w") as file:
    json.dump(augmented_dataset_and_information, file, indent=4)