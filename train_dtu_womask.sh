#! /bin/sh

output='dtu_scan105 dtu_scan106 dtu_scan110 dtu_scan114 dtu_scan118 dtu_scan122 dtu_scan24 dtu_scan37 dtu_scan40 dtu_scan55 dtu_scan63 dtu_scan65 dtu_scan69 dtu_scan83 dtu_scan97'


for item in $output; do
   echo "Start Training on" "$item" 
   python launch.py --config configs/model-a-dtu.yaml --gpu 5 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="womask20k"
   sleep 1
   python launch.py --config configs/model-b-dtu.yaml --gpu 5 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="womask20k"
   sleep 1
   python launch.py --config configs/model-c-dtu.yaml --gpu 5 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="womask20k"
   sleep 1
   python launch.py --config configs/neus-dtu.yaml --gpu 5 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="womask20k"
   sleep 1
done
