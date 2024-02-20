#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DragScene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from gaussian_renderer.calc_2d_coord import transform_points, calculate_depths, process_coordinates, find_deepest_3d_points
from gaussian_renderer.draw_points import draw_2ddrag_point
from PIL import Image
import sys

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_masked_gaussian(model_path, name, iteration, views, gaussians, pipeline, background):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masked_renders")

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        # load pre-trained gaussian model while building the scene
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def render_masked_gaussians(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        masked_gaussians = GaussianModel(dataset.sh_degree)

        masked_scene = Scene(dataset, gaussians=masked_gaussians, load_iteration=iteration, shuffle = False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_masked_gaussian(dataset.model_path, "masked", masked_scene.loaded_iter, masked_scene.getTrainCameras(), masked_gaussians, pipeline, background)

def pts_transform(dataset : ModelParams, iteration : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = DragScene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        means3D = gaussians.get_xyz
        points = means3D

        image_height = 800
        image_width = 800

        views = scene.getTransformCameras()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        start_points = [(400, 400)]
        end_points = [(600, 400)]

        start_points_3d = []
        end_points_3d = []
            
    # assume first view is source view, the following four views are projected views
    for idx, view in enumerate(tqdm(views, desc="Point transforming process")):
        
        if idx==0:
            print("source view")
            print("world_view_transform:", view.world_view_transform)
            print("full_proj_transform:", view.full_proj_transform)

            world_view_transform = view.world_view_transform
            full_proj_transform = view.full_proj_transform

            # render source image
            rendering = render(view, gaussians, pipeline, background)["render"] # rendered image
            image_path = os.path.join(dataset.model_path, "source&projection_images", '{0:05d}.png'.format(idx))
            torchvision.utils.save_image(rendering, image_path)
            image_variable = Image.open(image_path)
            draw_2ddrag_point(idx, image=image_variable, start_points=start_points, end_points=end_points, image_path=os.path.join(dataset.model_path, "source&projection_images"))

            # 2d projected position of all the 3d points
            # [[a1, b1], [a2, b2], ...]
            screen_coords = transform_points(points, world_view_transform, full_proj_transform, image_height, image_width)

            # [a, b]: [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ...]
            pixel_to_3dpoints = process_coordinates(screen_coords, points, image_height, image_width)

            pixel_to_3dpoints_str = '\n'.join(f'{k}: {v}' for k, v in pixel_to_3dpoints.items())
            file_path = './pixel_to_3dpoints.txt'
            with open(file_path, 'w') as file:
                file.write(pixel_to_3dpoints_str)

            # calculate the depth of each 3d points under current camera pose
            depths_3dpoints = calculate_depths(points, world_view_transform, full_proj_transform)

            # find the front 3d point of 2d images
            deepest_3dpoints_per_pixel = find_deepest_3d_points(pixel_to_3dpoints, depths_3dpoints)


            for point in start_points:
                point_3d = deepest_3dpoints_per_pixel.get(point)
                if point_3d:
                    start_points_3d.append(point_3d)
                    print("3d start point:", point_3d)
                else:
                    start_points_3d.append(None)

            for point in end_points:
                point_3d = deepest_3dpoints_per_pixel.get(point)
                if point_3d:
                    end_points_3d.append(point_3d)
                    print("3d end point:", point_3d)
                else:
                    end_points_3d.append(None)

            
            start_points_3d_default = [point if point is not None else (0, 0, 0) for point in start_points_3d]
            start_points_3d_tensor = torch.tensor(start_points_3d_default, dtype=torch.float)
            end_points_3d_default = [point if point is not None else (0, 0, 0) for point in end_points_3d]
            end_points_3d_tensor = torch.tensor(end_points_3d_default, dtype=torch.float)

            if torch.cuda.is_available():
                device = torch.device("cuda") 
                start_points_3d_tensor = start_points_3d_tensor.to(device)
                end_points_3d_tensor = end_points_3d_tensor.to(device)
            else:
                print("CUDA is not available. Tensors of 3d drag points are on CPU.")

        else:
            # print("projected view:", idx)
            # print("world_view_transform:", view.world_view_transform)
            # print("full_proj_transform:", view.full_proj_transform) 


            world_view_transform = view.world_view_transform
            full_proj_transform = view.full_proj_transform

            start_points_projected = transform_points(start_points_3d_tensor, world_view_transform, full_proj_transform, image_height, image_width)
            end_points_projected = transform_points(end_points_3d_tensor, world_view_transform, full_proj_transform, image_height, image_width)

            # render projected image
            rendering = render(view, gaussians, pipeline, background)["render"] # rendered image
            image_path = os.path.join(dataset.model_path, "source&projection_images", '{0:05d}.png'.format(idx))
            torchvision.utils.save_image(rendering, image_path)
            image_variable = Image.open(image_path)
            draw_2ddrag_point(idx, image=image_variable, start_points=start_points_projected, end_points=end_points_projected, image_path=os.path.join(dataset.model_path, "source&projection_images"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mask", type=bool, default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # pts_transform(model.extract(args), args.iteration)
    if args.render_mask:
        render_masked_gaussians(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    else:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)