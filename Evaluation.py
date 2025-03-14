import json
import numpy as np
from numpy.linalg import det, svd, qr

from utils import EulerAngles

with open('predictedMatrix.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

selection = 4

total_error_det = []
total_error_pred = []
total_error_orto = []
total_error = []

for i in range(len(data)):
    predicted_matrix = np.array(data[i]['predicted'])
    gt_matrix = np.array(data[i]['ground_truth'])
    determinant_predicted = det(predicted_matrix)
    determinant_gt = det(gt_matrix)
    orthogonal_predicted = predicted_matrix.T @ predicted_matrix
    orthogonal_gt = gt_matrix.T @ gt_matrix
    MSE_det = (determinant_predicted - 1) ** 2
    MSE_orto = np.mean((orthogonal_predicted - np.eye(3)) ** 2)
    MSE_pred = np.mean((predicted_matrix - gt_matrix) ** 2)
    match selection:
        case 0:
            print(f'Determinant z predikované matice: {np.round(determinant_predicted, 3)}')
            print(f'Determinant z originální matice: {np.round(determinant_gt, 3)}')
            print('Ověření ortognality:')
            print('Predikováná matice:')
            print(np.round(orthogonal_predicted, 2))
            print('Originální matice:')
            print(np.round(orthogonal_gt, 2))
        case 1:
            print('=====================================================')
            print(f'Chyba determinantu: {np.round(MSE_det, 6)}')
            print(f'Chyba ortogonality: {np.round(MSE_orto, 6)}')
            print(f'Chyba predikce:     {np.round(MSE_pred, 6)}')
            print(f'Celková chyba:      {np.round(MSE_pred + MSE_det + MSE_orto, 6)}')
        case 2:
            total_error_det.append(MSE_det)
            total_error_pred.append(MSE_pred)
            total_error_orto.append(MSE_orto)
            total_error.append(MSE_det + MSE_pred + MSE_orto)
        case 3:
            print(EulerAngles(gt_matrix, 'zyx', 5))
            print(EulerAngles(predicted_matrix, 'zyx', 5))
        case 4:
            U, S, Vt = svd(predicted_matrix)
            Q, R = qr(predicted_matrix)
            print('=========================================')
            print(i)
            print('gt:', EulerAngles(gt_matrix, 'zyx', 5))
            print('pred:', EulerAngles(predicted_matrix, 'zyx', 5))
            print('pred:', EulerAngles(U@Vt, 'zyx', 5))
            print(gt_matrix)
            print(U @ Vt)
            print((U @ Vt).T @ (U @ Vt))


match selection:
    case 2:
        print(np.mean(np.array(total_error_det)))
        print(np.mean(np.array(total_error_pred)))
        print(np.mean(np.array(total_error_orto)))
        print(np.mean(np.array(total_error)))