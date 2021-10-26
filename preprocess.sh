#!/bin/sh
printf "Generate detection training data  ...\n"
python ./preprocess/gen_train_det_data.py
printf "Crop character image  ...\n"
python ./preprocess/gen_train_rec_data_1.py
printf "Generate null character image ...\n"
python ./preprocess/gen_train_rec_data_2.py
printf "Generate fake character image ...\n"
python ./preprocess/gen_train_rec_data_3.py
printf "Train Dev split (character) ...\n"
python ./preprocess/gen_train_rec_data_4.py
printf "Train Dev split (null) ...\n"
python ./preprocess/gen_train_rec_data_5.py
printf "Finished!\n"