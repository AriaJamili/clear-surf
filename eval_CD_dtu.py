#!/usr/bin/env python3

#get chamfer distance comparisons on DTU between gt_mesh, ours
# It assumes we have created our meshes using create_my_meshes.py and they are at PACKAGE_ROOT/results/output_permuto_sdf_meshes/

####Call with######
# ./permuto_sdf_py/experiments/evaluation/evaluate_chamfer_distance.py --comp_name comp_1  [--with_mask]




import torch
import torchvision

import sys
import os
import numpy as np
import time
import argparse

#from datasets.dtu import DTUDatasetBase
import scripts.list_of_training_scenes as list_scenes
from scripts import mesh_filtering

import numpy as np
import cv2 as cv
from glob import glob
from scipy.io import loadmat
import trimesh
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
import multiprocessing as mp
import cv2
import subprocess
from PIL import Image


torch.manual_seed(42)
torch.set_default_tensor_type(torch.cuda.FloatTensor)




def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def eval(in_file, scene, dataset_dir, eval_dir, suffix=""):
    data_mesh = o3d.io.read_triangle_mesh(str(in_file))

    data_mesh.remove_unreferenced_vertices()

    mp.freeze_support()

    # default dtu values
    max_dist = 20
    patch = 60
    thresh = 0.2  # downsample density

    pbar = tqdm(total=8)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    #pbar.update(1)
    #pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    #rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    #mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    #for curr, idxs in enumerate(rnn_idxs):
    #    if mask[curr]:
    #        mask[idxs] = 0
    #        mask[curr] = 1
    #data_down = data_pcd[mask]
    data_down = data_pcd
    
    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{dataset_dir}/ObsMask/ObsMask{scene}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{dataset_dir}/Points/stl/stl{scene:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{dataset_dir}/ObsMask/Plane{scene}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = 1
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    B = np.array([[0, 0, 1]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[np.where(inbound)[0][grid_inbound][in_obs]] = R * data_alpha + W * (1 - data_alpha)
    data_color[np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:, 0] >= max_dist]] = G
    write_vis_pcd(f'{eval_dir}/vis_{scene:03}_d2s{suffix}.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
    stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G
    write_vis_pcd(f'{eval_dir}/vis_{scene:03}_s2d{suffix}.ply', stl, stl_color)

    pbar.update(1)
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    with open(f'{eval_dir}/result{suffix}.txt', 'a') as f:
        f.write(f'scene: {mean_d2s} {mean_s2d} {over_all}\n')

    return [mean_d2s, mean_s2d, over_all]

class EvalResults:
    def __init__(self):
        self.nr_scenes_evaluated=0
        self.total_d2s=0
        self.total_s2d=0
        self.total_overall=0
        self.scene_nr2overall = {}
    
    def update(self, scene_nr, d2s, s2d, overall):
        self.total_d2s+=d2s
        self.total_s2d+=s2d
        self.total_overall+=overall
        self.nr_scenes_evaluated+=1

        # print("adding overall", overall)
        # print("before adding scene2nroverall is ", self.scene_nr2overall)
        self.scene_nr2overall[scene_nr]=overall

    def get_results_avg(self):
        mean_d2s = self.total_d2s/self.nr_scenes_evaluated
        mean_s2d = self.total_s2d/self.nr_scenes_evaluated
        mean_overall = self.total_overall/self.nr_scenes_evaluated

        return mean_d2s, mean_s2d, mean_overall

    def get_results_per_scene(self):
        # print("returningdict ", self.scene_nr2overall)
        # print("in this object the total overall is ", self.total_overall)
        # print("in this object the nr_scenes is ", self.nr_scenes_evaluated)

        return self.scene_nr2overall 

def run_eval(mesh_path, scene_nr, dataset_dir, output_path ):
    
    #get the path to the evaluation script which is in ./DTUeval-python/eval.py
    cur_file_path=os.path.dirname(os.path.abspath(__file__))
    eval_script_path=os.path.join(cur_file_path,"scripts/DTUeval-python/eval.py")

    
    output_eval=subprocess.check_output([
            "python3", 
            eval_script_path, 
            "--data", mesh_path, 
            "--scan", scene_nr, 
            "--mode", "mesh", 
            "--dataset_dir", dataset_dir, 
            "--vis_out_dir", output_path
            ],  shell=False)

    #parse the outputs in the 3 components of mean_d2s, mean_s2d, over_all
    #the format is b'1.031434859827024 1.5483653154552615 1.2899000876411426\n'

    #remove letters and split
    output_eval=output_eval.decode("utf-8")
    output_eval=output_eval.split()
    output_eval=[float(x) for x in output_eval] #get to list fo floats


    return output_eval

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

#meshes trained without mask are actually cleaned by the mask by this code 
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

def clean_mesh(old_mesh_path, scan, dataset_path, output_path, model_name):
    old_mesh = trimesh.load(old_mesh_path)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, scan, dataset_path)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    #new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    
    #meshes = new_mesh.split(only_watertight=False)
    #new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    cams = np.load( os.path.join(dataset_path, 'dtu_scan{}/cameras_sphere.npz'.format(scan)) )

    n_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1

    scale_mats_np = []

    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [cams['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    vertices = new_vertices *scale_mats_np[0][0, 0] + scale_mats_np[0][:3, 3][None]

    mesh = trimesh.Trimesh(vertices, new_faces)
    mesh_path = os.path.join(output_path, f'{model_name}-{scan}-cleaned.ply')
    mesh.export(mesh_path)

    return mesh_path

def clean_pcd(old_pcd_path, scan, dataset_path):
    old_pcd = trimesh.load(old_pcd_path)
    old_vertices = old_pcd.vertices[:]
    mask = clean_points_by_mask(old_vertices, scan, dataset_path)
    new_vertices = old_vertices[np.where(mask)]

    new_pcd = trimesh.points.PointCloud(new_vertices)

    return new_pcd

def run():

    dataset="dtu"
    low_res=False
    
    #argparse
    parser = argparse.ArgumentParser(description='Quantitative comparison')
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    parser.add_argument('-p', '--path', type=str, default='/gris/gris-f/homestud/ajamili/datasets/DTU',
                       help='DTU data set path')
    parser.add_argument('-m', '--model', type=str, default='NeuS-dtu',
                       help='DTU data set path')
    args = parser.parse_args()

    config_training="with_mask_"+str(args.with_mask)
    # if not args.with_mask:
        # config_training+="_clean" #we use the cleaned up mesh that we get after runnign clean_mesh.py

    #path of my meshes
    root= os.path.dirname(os.path.abspath(__file__))
    results_path=os.path.join(root, "results")
    my_meshes_path=os.path.join(results_path, dataset)
    #path for gt
    gt_meshes_path=os.path.join(args.path, 'Points/stl')
    #outputs path where to dump the results after the evaluation
    output_path=os.path.join(results_path,"output_eval_chamfer_dist",dataset,config_training)
    os.makedirs( output_path, exist_ok=True)

    print("Data Set Path: ", args.path)
    print("Data Set MesH Path: ", gt_meshes_path)
    print("Model Mesh Path: ", my_meshes_path)


    results_mine=EvalResults() 

    #scenes_list=list_scenes.datasets[dataset]

    scenes_list = ["dtu_scan63"]

    dataset_config= dotdict(dict(name ="dtu", root_dir = "??", cameras_file = "cameras_sphere.npz",img_downscale = 2, apply_mask = "false"))
    tmp_mesh_path=os.path.join(results_path,"tmp")
    os.makedirs( tmp_mesh_path, exist_ok=True)

    for scan_name in scenes_list:
        print("scan_name: ",scan_name)
        root_dir_scene = os.path.join(args.path, scan_name)
        dataset_config.root_dir = root_dir_scene
        scan_nr=int(scan_name.replace('dtu_scan', ''))

        #get my mesh for this scene
        #my_mesh_path=os.path.join(my_meshes_path, args.model+"-"+scan_name+".obj")
        #my_mesh_path = "/gris/gris-f/homestud/ajamili/thesis-model/instant-nsr-pl/results/tmp/NeuS-dtu-dtu_scan24-cleaned.ply"
        my_mesh_path = "/gris/gris-f/homestud/ajamili/thesis-model/instant-nsr-pl/exp/Model-C-dtu-dtu_scan63/@20230707-062520/save/Model-C-dtu-dtu_scan63-it10000.obj"
        print("my_mesh_path is ", my_mesh_path)
        if not os.path.isfile(my_mesh_path):
            print("######COULD NOT FIND:  ", my_mesh_path)
            exit(1)
        else:
            
            my_mesh_path=clean_mesh(my_mesh_path, scan_nr,  args.path, tmp_mesh_path, args.model)
            #scene_nr_filled=str(scan_nr).zfill(3)
        
            #run DTU evaluation 
            # https://github.com/jzhangbs/DTUeval-python
            output_eval_my_mesh = eval(my_mesh_path,scan_nr,args.path,output_path,suffix="Model-A")
            print("output_eval_my_mesh", output_eval_my_mesh)
            results_mine.update(scan_nr, output_eval_my_mesh[0], output_eval_my_mesh[1], output_eval_my_mesh[2])


      

    #finished reading all scenes
    #print results
    ####MINE
    print("---------MINE--------")
    #mine_avg=results_mine.get_results_avg() 
    #mine_per_scene=results_mine.get_results_per_scene()
    #print("mine_avg_overall", mine_avg[2])
    #print("mine_per_scene", mine_per_scene)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def scale_anything(d):
    r = -(d[1:] + d[0:]) /2
    return r[0]

def main():
    run()



if __name__ == "__main__":
     main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
