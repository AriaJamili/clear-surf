#! /bin/sh

output='scan105 scan106 scan110 scan114 scan118 scan122 scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97'


for item in $output; do
   echo "Start Training on" "$item" 
   python launch.py --config configs/model-a-dtu_bg.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
   python launch.py --config configs/model-b-dtu_bg.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
   python launch.py --config configs/model-c-dtu_bg.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
   python launch.py --config configs/neus-dtu.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
done
