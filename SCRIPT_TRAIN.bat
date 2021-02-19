python keras_retinanet/bin/train.py --epochs 5 --steps 129 csv data/annotations.csv data/classes.csv
python keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_05.h5 snapshots/model.h5
pause