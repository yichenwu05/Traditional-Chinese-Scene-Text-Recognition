#!/bin/sh
printf "Runinng Step 1 ...\n"
python ./inference/step1_croptextarea.py $1
printf "Step 1 Finished!\n\n"
printf "Runinng Step 2 ...\n"
python ./inference/step2_detectchar.py $1
printf "Step 2 Finished!\n\n"
printf "Runinng Step 3 ...\n"
python ./inference/step3_char_cls.py $1
printf "Runinng Step 4 ...\n"
python ./inference/step4_nlp_adjust.py $1
printf "Step 4 Finished!\n"