import os
import numpy as np
import json

import torch

from opts import parse_opts
from dataLoader import NiftiDataset
from predict_planes import test

path = r'D:\DataSet\Data\model34_ao.pth'

model = torch.load(path)

opt = parse_opts()
opt.dir = r'D:\Dataset\Data'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform_info_file_train = os.path.join(opt.dir, 'trainInfo.json')

train_dataset = NiftiDataset(root_dir=opt.dir, transform_info_file=transform_info_file_train, resize=opt.resize)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

avg_loss, results = test(model, train_loader, device)

with open('predictedMatrixTrain.json', 'w', encoding='utf-8') as file:
    json.dump(results, file, indent=4, ensure_ascii=False)