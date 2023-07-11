#!/usr/bin/env python3

#Clean DTU gt_mesh



import torch
import torchvision

import os
import numpy as np
import argparse
import cv2 as cv
from glob import glob
from tqdm import tqdm
import trimesh


torch.manual_seed(42)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


#meshes are cleaned by the mask by this code 
# https://github.com/Totoro97/NeuS/issues/74
def clean_points_by_mask(points, scan, dataset_path):
    cameras = np.load( os.path.join(dataset_path, 'dtu_scan{}/cameras_sphere.npz'.format(scan))  )
    mask_lis = sorted(glob( os.path.join(dataset_path, 'dtu_scan{}/mask/*.png'.format(scan))   ))

    n_images = 49 if scan < 83 else 64
    inside_mask = np.ones(len(points)) > 0.5
    for i in range(n_images):
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (101, 101))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        inside_mask &= curr_mask.astype(bool)

    return inside_mask

def clean_pcd(old_pcd_path, scan, dataset_path, output_path):
    pbar = tqdm(total=3)
    pbar.set_description('read gt pcd')
    
    old_pcd = trimesh.load(old_pcd_path)
    old_vertices = old_pcd.vertices[:]
    old_color = old_pcd.colors[:]

    pbar.update(1)
    pbar.set_description('cleaning pcd')
    mask = clean_points_by_mask(old_vertices, scan, dataset_path)
    new_vertices = old_vertices[np.where(mask)]
    new_color = old_color[np.where(mask)]

    pbar.update(1)
    pbar.set_description('exporting pcd')
    new_pcd = trimesh.points.PointCloud(new_vertices, colors=new_color)
    mesh_path = os.path.join(output_path, f'stl{scan:03}_total-cleaned.ply')
    new_pcd.export(mesh_path)
    pbar.update(1)
    pbar.close()

def run():

    dataset="dtu"
    
    #argparse
    parser = argparse.ArgumentParser(description='Clean DTU PCD for CD')
    parser.add_argument('-p', '--path', type=str, default='/gris/gris-f/homestud/ajamili/datasets/DTU',
                       help='DTU data set path')
    args = parser.parse_args()



    #paths
    root= os.path.dirname(os.path.abspath(__file__))
    results_path=os.path.join(root, "eval")
    output_path=os.path.join(results_path, dataset, 'stl')
    gt_meshes_path=os.path.join(args.path, 'Points/stl')
    os.makedirs( output_path, exist_ok=True)

    print("Data Set Path: ", args.path)
    print("Data Set MesH Path: ", gt_meshes_path)
    print("Output Mesh Path: ", output_path)


    scenes_list=["dtu_scan105","dtu_scan106", "dtu_scan110", "dtu_scan114", "dtu_scan118", "dtu_scan122", "dtu_scan24", "dtu_scan37", "dtu_scan40", "dtu_scan55", "dtu_scan63", "dtu_scan65", "dtu_scan69", "dtu_scan83", "dtu_scan97"]


    for scan_name in tqdm(scenes_list, desc=f"Cleaning", unit="scan") :
        print("scan_name: ",scan_name)
        scan_nr=int(scan_name.replace('dtu_scan', ''))
        gt_mesh_path=os.path.join(gt_meshes_path, f'stl{scan_nr:03}_total.ply')
        if not os.path.isfile(gt_mesh_path):
            print("######COULD NOT FIND:  ", gt_mesh_path)
            exit(1)
        else:
            clean_pcd(gt_mesh_path,scan_nr,args.path, output_path)
            

def main():
    run()



if __name__ == "__main__":
     main() 
