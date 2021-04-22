import os
import sys


FILE_SUFFIX = '_datamap.png'
CLS_NAMES = ['plate', 'thruster', 'circlet', 'slat', 'part']
          
def join_labels(dataset_type):
    path = 'data/' + dataset_type + '/'
    datasets = [dataset for dataset in os.listdir(path) if os.path.isdir(path + dataset)]
     
    with open('data/annotations.csv', 'w') as labels:
        for dataset in datasets:
            dataset_path = path + '/' + dataset
            for file in os.listdir(dataset_path):
                if '.txt' in file:
                    name = file.split('.')[0]
                    print('processing: ' + name)
                    
                    with open(dataset_path + '/' + file, 'r') as single_labels:
                        line = single_labels.readline()
                        while line:
                            coords = line.split(',')
                            if len(coords) == 5:
                                x_min, y_min, x_max, y_max, cls = [int(c) for c in coords]
                                
                                print(dataset_type + '/' + dataset + '/' + name + FILE_SUFFIX, file=labels, end=',')
                                print('{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, CLS_NAMES[cls]), file=labels)
                            line = single_labels.readline()         
                        

if __name__ == '__main__':
    if len(sys.argv) == 1:
        join_labels('train')
    elif len(sys.argv) == 2:
        join_labels(sys.argv[1])