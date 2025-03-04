import random
import json
import os


def SplitInfo(directory_list, training_size=0.7, validation_size=0.1, testing_size=0.2):

    FolderIDs = []

    for data in directory_list:
        FolderID = data['ID']['OriginalID']
        if FolderID in FolderIDs:
            continue
        else:
            FolderIDs.append(FolderID)

    random.shuffle(FolderIDs)

    subset1 = FolderIDs[:int(len(FolderIDs) * training_size)]
    subset2 = FolderIDs[int(len(FolderIDs) * training_size):
                        int(len(FolderIDs) * training_size)+int(len(FolderIDs) * validation_size)]
    subset3 = FolderIDs[int(len(FolderIDs) * training_size)+int(len(FolderIDs) * validation_size):]

    return [subset1, subset2, subset3]


def GenerateInfo(subset_list, directory_list, path):
    for i in range(len(subset_list)):
        subsetInfo = []
        for j in range(len(directory_list)):
            if directory_list[j]['ID']['OriginalID'] in subset_list[i]:
                subsetInfo.append(directory_list[j])
        match i:
            case 0:
                file_name = 'trainInfo.json'
            case 1:
                file_name = 'validInfo.json'
            case 2:
                file_name = 'testInfo.json'

        with open(os.path.join(path, file_name), 'w') as file:
            json.dump(subsetInfo, file, indent=4)

    return 1


root_dir = r'D:\DataSet'
path = os.path.join(root_dir, 'Info.json')

with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

subsets = SplitInfo(data)
GenerateInfo(subsets, data, os.path.join(root_dir, 'Data'))