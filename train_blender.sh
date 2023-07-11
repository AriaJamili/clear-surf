#! /bin/sh


output='scene_1 scene_2 scene_3 scene_4 scene_5 scene_6 scene_7 scene_8 scene_9'


for item in $output; do
   echo "Start Training on" "$item"  # remember quotes here
   python launch.py --config configs/model-a-blender.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
   python launch.py --config configs/model-b-blender.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
   python launch.py --config configs/model-c-blender.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
   python launch.py --config configs/neus-blender.yaml --gpu 5 --train dataset.scene=$item
   sleep 1
done

