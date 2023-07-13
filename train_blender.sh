#! /bin/sh


output='scene_0 scene_1 scene_2 scene_3 scene_4 scene_5 scene_6 scene_7 scene_8 scene_9'

for item in $output; do
   echo "Start Training on" "$item"  # remember quotes here
   python launch.py --config configs/model-a-blender-wmask.yaml --gpu 2 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="wmask20k"
   sleep 1
   python launch.py --config configs/model-b-blender-wmask.yaml --gpu 2 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="wmask20k"
   sleep 1
   python launch.py --config configs/model-c-blender-wmask.yaml --gpu 2 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="wmask20k"
   sleep 1
   python launch.py --config configs/neus-blender-wmask.yaml --gpu 2 --train dataset.scene=$item model.cos_anneal_end=20000 model.train_num_rays=1024 trial_name="wmask20k"
   sleep 1
done

