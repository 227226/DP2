def Slicer3D(img3D, axis, ax, coor_system):

    """
    Function is used to show 3D data with the option of scrolling along a chosen axis.

    Inputs:
    :param img3D: (nd.array) - converted image that should be visualized (should be a 3D array)
    :param axis: (str) - 'axial', 'sagital', or 'coronal' for chosen axis
    :param ax:  (plt.Axes) - axis from plt.subplots(), specific axis that should be connected with projected image

    Outputs:
    :return: - visualization with the option of scrolling over image slices
    """

    # import of essential packages:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton
    from scipy.ndimage import rotate

    if coor_system == 'Original':
        g_name = ['Axiální rovina', 'Sagitální rovina', 'Koronální rovina']
    elif coor_system == 'SA':
        g_name = ['pseudoVLA', 'SA', 'PseudoHLA']

    # adjustment of view plane:
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
        current_frame = current_frame
        num_frames = img3D.shape[1]
        slice_func = lambda i: img3D[:, i, :]
        plain = g_name[2]

    # first image adjustment:
    frame_img = ax.imshow(slice_func(current_frame), cmap='gray')
    ax.set_title(f'{plain} - snímek: {current_frame + 1}/{num_frames}')
    ax.axis('off')

    # definition of event connected with scrolling:
    def on_scroll(event):
        nonlocal current_frame
        if event.inaxes == ax:
            if event.button == 'up':
                current_frame = (current_frame + 1) % num_frames
            elif event.button == 'down':
                current_frame = (current_frame - 1) % num_frames

            # actualisation of current frame
            frame_img.set_data(slice_func(current_frame))
            ax.set_title(f'{plain} - snímek: {current_frame + 1}/{num_frames}')
            ax.figure.canvas.draw_idle()

    # connection of canvas and scroll event:
    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)


def RotationAffine3D(img, A, center_point=None):
    """
    Function performs affine rotation of image using a specified affine matrix.
    Rotation is performed around a given center.

    Inputs
    :param img: (nd.array) - 3D image array
    :param A: (nd.array) - affine rotation matrix
    :param center_point: (tuple or None) - center of rotation (z, y, x).
                         If None, the image center is used.

    Outputs
    :return: img_rot (nd.array) - 3D rotated image array
    """

    import numpy as np
    from scipy.ndimage import affine_transform

    # Default center is the image center if not provided
    if center_point is None:
        center_point = (np.array(img.shape) - 1) / 2
    else:
        center_point = np.array(center_point)

    # Offset calculations for rotation around the desired center
    offset_to_center = -center_point

    # Perform the affine transformation with the correct offset
    img_rot = affine_transform(
        img,
        matrix=A[:3, :3],
        offset=np.dot(A[:3, :3], offset_to_center) - offset_to_center,
        order=3,
        mode='constant',
        cval=0
    )

    return img_rot


def LoadniiAndPad(nii_data, padding=(0, 0, 0)):

    """
    Function loads nii file into numpy array and in case of non-zero padding in some axis it adds zeros to this axis.

    Inputs:
    :param nii_data: (nib.file) - nifti data containing header and array
    :param padding: (tuple) - number of zeros that should be added in chosen axis

    Outputs:
    :return new_img: (nd.array) - image with or without padding in numpy format
    """

    import numpy as np

    # loading of image data from nifti structure:
    img_data = nii_data.get_fdata()
    # number of zeros that should be added in given dimensions:
    pad_1, pad_2, pad_3 = padding
    # calculation of boundaries indexes:
    orig_1, orig_2, orig_3 = np.array(img_data.shape) // 2

    # creation of new image, into this image original image will be given from the center
    new_img = np.zeros((img_data.shape[0] + pad_1,
                        img_data.shape[1] + pad_2,
                        img_data.shape[2] + pad_3))
    # calculation of image center
    center_1, center_2, center_3 = np.array(new_img.shape) // 2

    # giving the original image into created image of zeros:
    new_img[center_1 - orig_1: center_1 + orig_1,
            center_2 - orig_2: center_2 + orig_2,
            center_3 - orig_3: center_3 + orig_3] = img_data

    # returns padded image:
    return new_img


def RestrictImg(img, factors=(0.4, 0.4, 0.3)):

    """
    Function executes restriction of image according to given factors by default or by user.
    Restriction is performed around image center.

    Inputs:
    :param img: (nd.array) - 3D image array for restriction
    :param factors: (tuple) - gives percentage of image to preserve

    Outputs:
    :return: restricted_img (nd.array) - 3D restricted image
    """

    # extraction of factors:
    x_comp = factors[0]
    y_comp = factors[1]
    z_comp = factors[2]

    # calculation of 3D image shape:
    center_x = img.shape[0] // 2
    center_y = img.shape[1] // 2
    center_z = img.shape[2] // 2

    # calculation of 3D image center:
    x_size = int((img.shape[0] * x_comp) // 2)
    y_size = int((img.shape[1] * y_comp) // 2)
    z_size = int((img.shape[2] * z_comp) // 2)

    # restriction of image:
    restricted_img = img[center_x-x_size:center_x+x_size,
                            center_y-y_size:center_y+y_size,
                            center_z-z_size:center_z+z_size]

    # returns restricted image around the center:
    return restricted_img


def GaussianNoise3D(img, factor):

    """
    Function serves as generator of 3D Gaussian noise.

    Inputs:
    :param img: (nd.array) - original 3D image array
    :param factor: (float) - multiplication factor

    Outputs:
    :return noisy_img: (nd.array) - image with additional Gaussian noise
    """

    import numpy as np

    # parameters of noise:
    mean = 0
    std_dev = factor * np.max(img)

    # generation of noise:
    gaussian_noise = np.random.normal(loc=mean, scale=std_dev, size=img.shape)

    # addition of noise into the image:
    noisy_img = img + gaussian_noise

    # clipping of image values:
    noisy_img = np.clip(noisy_img, np.min(img), np.max(img))

    # returns noise with additional 3D Gaussian noise:
    return noisy_img


def AffineMatrix(angles=(0, 0, 0), order_of_rotation='xyz', scale=(1, 1, 1), scale_order=None):

    """
    Function perform calculation of affine rotation matrix in order set by user. Beside simple rotation
    there is also possibility to add scaling into the matrix.

    inputs:
    :param angles: (tuple) -  defines angles of rotation around x-, y-, and z-axis in this order
    :param order_of_rotation: (string) - defines order of application of rotation simple matrices around
                                         x-, y- and z-axis, there are six possibilities:
                                         1. xyz (default), 2. xzy, 3. yxz, 4.yzx, 5.zxy, 6. zyx
    :param scale: (tuple) - defines scaling in x-, y- and z-axis
    :param scale_order: (NoneType, string) - defines if scaling is used on affine rotation matrix
                                           - 3 options:
                                             1. if None no scaling is applied (default)
                                             2. if before scaling is applied before rotation
                                             3. if after scaling is applied after rotation
    outputs:
    :return: - A_final (nd.array) - affine rotation matrix with or without scaling
    """

    import numpy as np

    # transform from degrees to radians:
    alpha, beta, gamma = np.deg2rad(np.array(angles))

    # definition of scaling matrix:
    A_scale = np.array([
                        [scale[0], 0, 0],
                        [0, scale[1], 0],
                        [0, 0, scale[2]]
                       ])
    # definition of rotation around x-axis:
    Ax = np.array([
                   [1,             0,              0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha),  np.cos(alpha)]
                  ])
    # definition of rotation around y-axis:
    Ay = np.array([
                   [ np.cos(beta),  0,   np.sin(beta)],
                   [ 0,             1,              0],
                   [-np.sin(beta),  0,   np.cos(beta)]
                  ])
    # definition of rotation around z-axis:
    Az = np.array([
                   [np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma),  np.cos(gamma), 0],
                   [            0,              0, 1]
                  ])

    # definition of rotation order given by user or default settings:
    if order_of_rotation == 'xyz':
        A = Az @ Ay @ Ax
    elif order_of_rotation == 'xzy':
        A = Ay @ Az @ Ax
    elif order_of_rotation == 'yxz':
        A = Az @ Ax @ Ay
    elif order_of_rotation == 'yzx':
        A = Ax @ Az @ Ay
    elif order_of_rotation == 'zxy':
        A = Ay @ Ax @ Az
    elif order_of_rotation == 'zyx':
        A = Ax @ Ay @ Az

    # final affine matrix calculation, scaling is applied according to users specifications or
    # not if None in default:
    if scale_order == 'before':
        A_final = A @ A_scale
    elif scale_order == 'after':
        A_final = A_scale @ A
    elif scale_order == None:
        A_final = A

    # final affine matrix as output of the function:
    return A_final


def EulerAngles(A, order_of_rotation='xyz', decimals=5):

    """
    Function decompose affine matrix A into Euler angles, in given order with specified precision in
    decimals.

    Inputs:
    :param A: (nd.array) - affine matrix, should be rotation matrix in other case there will be no
                           correct results
    :param order_of_rotation (string) - specifies order of decomposition, there are six possibilities
                                        of decomposition order:
                                        1. xyz (default), 2. xzy, 3. yxz, 4. yzx, 5. zxy, 6. zyx
    :param decimals (int) - specifies number of decimals in resulting angle (precision)

    Outputs:
    :return: angles (nd.array) - output array with resulting angles in order of rotation  in x-, y-
                                 and z-axis
    """

    import numpy as np

    # decomposition of affine matrix into single numbers for following calculations:
    A11, A12, A13 = A[0, :]
    A21, A22, A23 = A[1, :]
    A31, A32, A33 = A[2, :]

    # estimation of angles given by order of decomposition, default order 'xyz':
    if order_of_rotation == 'xyz':
        alpha = np.arctan2(A32, A33)
        beta = np.arctan2(-A31, np.sqrt(A21 ** 2 + A11 ** 2))
        gamma = np.arctan2(A21, A11)
    elif order_of_rotation == 'xzy':
        alpha = np.arctan2(-A23, A22)
        beta = np.arctan2(-A31, A11)
        gamma = np.arctan2(A21, np.sqrt(A22 ** 2 + A23 ** 2))
    elif order_of_rotation == 'yxz':
        alpha = np.arctan2(A32, np.sqrt(A31 ** 2 + A33 ** 2))
        beta = np.arctan2(-A31, A33)
        gamma = np.arctan2(-A12, A22)
    elif order_of_rotation == 'yzx':
        alpha = np.arctan2(A32, A22)
        beta = np.arctan2(A13, A11)
        gamma = np.arctan2(-A12, np.sqrt(A22 ** 2 + A32 ** 2))
    elif order_of_rotation == 'zxy':
        alpha = np.arctan2(-A23, np.sqrt(A21 ** 2 + A22 ** 2))
        beta = np.arctan2(A13, A33)
        gamma = np.arctan2(A21, A22)
    elif order_of_rotation == 'zyx':
        alpha = np.arctan2(-A23, A33)
        beta = np.arctan2(A13, np.sqrt(A23 ** 2 + A33 ** 2))
        gamma = np.arctan2(-A12, A11)

    # creation of array with resulting angles in order of rotation around x-, y-, z-axis
    angles = np.array([np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)])
    angles = np.round(angles, decimals)

    # final Euler angles in degrees as output of the function:
    return angles


def ScaleOriginalImageShape(orig_img, z_pad, A):

    """
    Function scales original image corner points.

    Inputs:
    :param orig_img: (nd.array) - image
    :param z_pad (int) - number of padded zeros
    :param A (nd.array) - scaling matrix A

    Outputs:
    :return: shape (list) - maxima point location
    """

    import numpy as np

    # extraction of original shape:
    orig_shape = np.array(orig_img.shape)
    # removing of padding:
    orig_shape[2] -= z_pad
    # new shape calculation:
    new_shape = A @ orig_shape

    return [int(new_shape[0]), int(new_shape[1]), int(new_shape[2])]


def CreateCube(orig_img, padded_img):

    """
    Function creates cube specified by corner points.

    Inputs:
    :param orig_img: (nd.array) - original image
    :param padded_img (nd.array) - padded image if padding is done, if not original image should be put in again

    Outputs:
    :return: points (nd.array) - corner coordinates of image volume
    :return: faces (list) - faces of image volume
    """

    import numpy as np

    # coordinates for specification of corner points:
    x0, y0, z0 = orig_img.shape
    x1, y1, z1 = padded_img.shape

    dx = (x1 - x0) / 2
    dy = (y1 - y0) / 2
    dz = (z1 - z0) / 2
    # points definition:
    points = np.array([
                       [ 0,  0,  0],
                       [ 0,  0, z0],
                       [x0,  0, z0],
                       [x0,  0,  0],
                       [ 0, y0,  0],
                       [ 0, y0, z0],
                       [x0, y0, z0],
                       [x0, y0,  0],
                                    ], dtype=np.float64)
    # faces definition:
    faces = [
             [points[0], points[1], points[2], points[3]], # přední stěna
             [points[4], points[5], points[6], points[7]], #  zadní stěna
             [points[0], points[4], points[5], points[1]], #   levá stěna
             [points[3], points[7], points[6], points[2]], #  pravá stěna
             [points[0], points[4], points[7], points[3]], #  dolní stěna
             [points[1], points[5], points[6], points[2]]  #  horní stěna
               ]
    # specification of position if padding is done:
    points[:, 0] += dx
    points[:, 1] += dy
    points[:, 2] += dz

    return points, faces


def CubeRotation(points, A, rotation_center=(0, 0, 0)):

    """
    Function performs rotation of cube corner points by specified rotation defined by rotation matrix A and rotation
    center.

    Inputs:
    :param points: (nd.array) - cube corner points
    :param A (nd.array) - rotation matrix
    :param rotation_center: (tuple) - position of rotation center

    Outputs:
    :return: rotated_points (nd.array) - corner coordinates of cube
    :return: rotated_faces (list) - faces of bounding box specified by corner points
    """

    import numpy as np

    # augmentation of points (1 is append):
    points_aug = np.zeros((points.shape[0], points.shape[1]+1))
    points_aug[:, 3] = 1
    points_aug[:, :3] = points

    # initialization of variable for storing rotated points:
    rotated_points = np.zeros(points_aug.shape)

    # creation of augmented rotation matrix:
    A0 = np.eye(4)
    A0[:3, :3] = A

    # augmented translation matrices:
    Tf = np.eye(4)
    Tb = np.eye(4)

    # adding of translation:
    rotation_center = np.array(rotation_center)
    Tf[:3, 3] = rotation_center
    Tb[:3, 3] = -rotation_center

    # final transform matrix:
    Af = Tf @ A0 @ Tb

    # rotation of points:
    for i in range(points.shape[0]):
        rotated_points[i, :] = Af @ points_aug[i, :]

    rotated_points = rotated_points[:, :3]

    rotated_faces = [
                     [rotated_points[0], rotated_points[1], rotated_points[2], rotated_points[3]], # front face
                     [rotated_points[4], rotated_points[5], rotated_points[6], rotated_points[7]], # back face
                     [rotated_points[0], rotated_points[4], rotated_points[5], rotated_points[1]], # left face
                     [rotated_points[3], rotated_points[7], rotated_points[6], rotated_points[2]], # right face
                     [rotated_points[0], rotated_points[4], rotated_points[7], rotated_points[3]], # lower face
                     [rotated_points[1], rotated_points[5], rotated_points[6], rotated_points[2]]  # upper face
                    ]

    return rotated_points, rotated_faces


def CreateBoundingBox(size=(200, 200, 150), center_point=(0, 0, 0)):

    """
    Function returns bounding box with specified size, respectively corner points of bounding box and its faces.

    Inputs:
    :param size: (tuple) - size of bounding box
    :param center_point (tuple) - specifies exact location of bounding box in coordinate system

    Outputs:
    :return: bb (nd.array) - corner coordinates of bounding box
    :return: bb_faces (list) - faces of bounding box specified by corner points
    """

    import numpy as np

    # specification of bounding box corner points:
    bb = np.array([
                              [0      ,       0,       0],
                              [0      ,       0, size[2]],
                              [size[0],       0, size[2]],
                              [size[0],       0,       0],
                              [0      , size[1],       0],
                              [0      , size[1], size[2]],
                              [size[0], size[1], size[2]],
                              [size[0], size[1],       0]
                                                         ], dtype=np.float64)
    # centering:
    bb[:, 0] += center_point[0] - size[0] / 2
    bb[:, 1] += center_point[1] - size[1] / 2
    bb[:, 2] += center_point[2] - size[2] / 2
    # faces definition:
    bb_faces = [
                [bb[0], bb[1], bb[2], bb[3]],
                [bb[4], bb[5], bb[6], bb[7]],
                [bb[0], bb[4], bb[5], bb[1]],
                [bb[3], bb[7], bb[6], bb[2]],
                [bb[0], bb[3], bb[7], bb[4]],
                [bb[1], bb[5], bb[6], bb[2]]
                                            ]
    return bb, bb_faces


def AnglesVariations():

    """
    Function returns all possible angle rotations along each axis according to permitted rotations.

    Outputs:
    :return: rotation variations (nd.array) - output variable with all possible angles variations driven by
                                                permitted rotations
    """

    import numpy as np
    import itertools

    # permitted rotations along each axis:
    angles_z = [0, 15, -15, 30, -30, 45, -45]  # Rotace kolem osy X
    angles_y = [0, 10, -10, 20, -20]  # Rotace kolem osy Y
    angles_x = [0, 10, -10, 20, -20]  # Rotace kolem osy Z
    # variation of all possible rotations:
    rotation_variations = list(itertools.product(angles_x, angles_y, angles_z))
    # conversion to nd.array
    rotation_variations = np.array(rotation_variations)

    return rotation_variations


def plane_from_points(P1, P2, P3):

    """
    Function calculates equation of plane from three specified points.

    Inputs:
    :param P1, P2, P3: (int) - points from exact plane

    Outputs:
    :return: A, B, C, D (float) - constant of plane equation
    """

    import numpy as np

    # Směrové vektory
    v1 = np.array(P2) - np.array(P1)
    v2 = np.array(P3) - np.array(P1)

    # Normálový vektor (vektorový součin)
    normal = np.cross(v1, v2)
    A, B, C = normal

    # Výpočet D ze vzorce roviny
    D = -np.dot(normal, P1)

    return A, B, C, D


def PermittedState(points, bb_points, angles=(0, 0, 0)):

    """
    Function checks if whole volume of bounding box lays inside the original rotated volume. If not (0, 0, 0)
    tuple is returned by the function and is filtered in the next block. Only non-zero rotations are proved.

    Inputs:
    :param points: (nd.array) - matrix with coordinate specifying corner locations of original image volume
    :param bb_points (nd.array) - corner coordinates of bounding box
    :param angles (tuple) - is also returned if the bounding box lays inside the original rotated volume

    Outputs:
    :return: angles (tuple) - output variable with angles rotation
    """

    # decomposition of array into single points:
    P1 = points[0, :]
    P2 = points[1, :]
    P3 = points[2, :]
    P4 = points[3, :]
    P5 = points[4, :]
    P6 = points[5, :]
    P7 = points[6, :]
    # P8 = points[7, :]

    # control is made by substitution of points into equation of plane, for each plane
    # there is specification where point of bounding box must lay, only if all conditions
    # are satisfied the angles are returned in other case only zeros are returned:
    # left side:
    A1, B1, C1, D1 = plane_from_points(P1, P2, P3)
    for i in range(bb_points.shape[0]):
        in_box = A1 * bb_points[i, 0] + B1 * bb_points[i, 1] + C1 * bb_points[i, 2] + D1
        if in_box > 0:
            continue
        else:
            return (0, 0, 0)

    # right side:
    A2, B2, C2, D2 = plane_from_points(P5, P6, P7)
    for i in range(bb_points.shape[0]):
        in_box = A2 * bb_points[i, 0] + B2 * bb_points[i, 1] + C2 * bb_points[i, 2] + D2
        if in_box < 0:
            continue
        else:
            return (0, 0, 0)

    # front side:
    A3, B3, C3, D3 = plane_from_points(P3, P4, P7)
    for i in range(bb_points.shape[0]):
        in_box = A3 * bb_points[i, 0] + B3 * bb_points[i, 1] + C3 * bb_points[i, 2] + D3
        if in_box < 0:
            continue
        else:
            return (0, 0, 0)

    # back side:
    A4, B4, C4, D4 = plane_from_points(P1, P2, P6)
    for i in range(bb_points.shape[0]):
        in_box = A4 * bb_points[i, 0] + B4 * bb_points[i, 1] + C4 * bb_points[i, 2] + D4
        if in_box < 0:
            continue
        else:
            return (0, 0, 0)

    # upper side:
    A5, B5, C5, D5 = plane_from_points(P2, P3, P6)
    for i in range(bb_points.shape[0]):
        in_box = A5 * bb_points[i, 0] + B5 * bb_points[i, 1] + C5 * bb_points[i, 2] + D5
        if in_box < 0:
           continue
        else:
           return (0, 0, 0)

    # lower side:
    A6, B6, C6, D6 = plane_from_points(P1, P4, P5)
    for i in range(bb_points.shape[0]):
        in_box = A6 * bb_points[i, 0] + B6 * bb_points[i, 1] + C6 * bb_points[i, 2] + D6
        if in_box > 0:
            continue
        else:
            return (0, 0, 0)

    return angles
