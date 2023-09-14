import argparse
import os
import sys
import time

import commentjson as json
import numpy as np
import wandb
from common import *  # noqa
from scenes import *  # noqa
from tqdm import tqdm

pyngp_path = "/cluster/home/jpostels/nnaimi/instant-ngp/build"
sys.path.append(pyngp_path)

import pyngp as ngp  # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run instant neural graphics primitives with additional configuration & output options"  # noqa
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.",  # noqa
    )

    parser.add_argument(
        "--scene",
        "--training_data",
        default="",
        help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.",  # noqa
    )
    parser.add_argument(
        "--network",
        default="",
        help="Path to the network config. Uses the scene's default if unspecified.",
    )

    parser.add_argument(
        "--load_snapshot",
        "--snapshot",
        default="",
        help="Load this snapshot before training. recommended extension: .ingp/.msgpack",
    )
    parser.add_argument(
        "--save_snapshot",
        default="",
        help="Save this snapshot after training. recommended extension: .ingp/.msgpack",
    )

    parser.add_argument(
        "--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of."
    )
    parser.add_argument(
        "--screenshot_dir", default="", help="Which directory to output screenshots to."
    )
    parser.add_argument(
        "--screenshot_spp",
        type=int,
        default=16,
        help="Number of samples per pixel in screenshots.",
    )

    parser.add_argument(
        "--width",
        "--screenshot_w",
        type=int,
        default=0,
        help="Resolution width of GUI and screenshots.",
    )
    parser.add_argument(
        "--height",
        "--screenshot_h",
        type=int,
        default=0,
        help="Resolution height of GUI and screenshots.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="If the GUI is enabled, controls whether training starts immediately.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=-1,
        help="Number of steps to train for before quitting.",
    )
    parser.add_argument(
        "--name", default="", help="name of run for weightsandbiases logging"
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="frequency with which to log training information",
    )
    parser.add_argument(
        "--log_imgs",
        action="store_true",
        help="logs images in addition to training information",
    )

    return parser.parse_args()


def get_scene(scene):
    for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:  # noqa
        if scene in scenes:
            return scenes[scene]
    return None


def update_config_file(filename):
    with open(args.network, "r+") as f:
        config = json.load(f)
        # config["encoding"]["n_features_per_level"] = wandb.config.get(
        #     "encoding.n_features_per_level"
        # )
        # config["encoding"]["log2_hashmap_size"] = wandb.config.get(
        #     "encoding.log2_hashmap_size"
        # )
        config["encoding"]["base_resolution"] = wandb.config.get(
            "encoding.base_resolution"
        )
        # config["encoding"]["n_levels"] = wandb.config.get("encoding.n_levels")
        # config["encoding"]["per_level_scale"] = wandb.config.get(
        #     "encoding.per_level_scale"
        # )
        f.seek(0)  # <--- reset file position to the beginning
        json.dump(config, f, indent=4)
        f.truncate()  # remove remaining part


def main(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run = wandb.init(
        name=f"{args.name}-{timestamp}",
        tags=["sweep", "base-res-rate-distortion"],
        entity="inrcompression",
        project="instantNGP-4-Compression",
        dir="/cluster/work/cvl/jpostels/nnaimi/wandb",
    )

    # overwrite the parameters with the ones provided by the sweep
    update_config_file(args.network)

    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR  # noqa

    # Open config JSON file and load its contents into a dictionary
    with open(args.network, "r") as json_file:
        config = json.load(json_file)

    # Load the PNG image (converts the image from srgb to linear np array)
    groundtruth = read_image(args.scene)  # noqa

    for file in args.files:
        print("files")
        scene_info = get_scene(file)
        if scene_info:
            file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
        testbed.load_file(file)

    if args.scene:
        scene_info = get_scene(args.scene)
        if scene_info is not None:
            args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
            if not args.network and "network" in scene_info:
                args.network = scene_info["network"]

        testbed.load_training_data(args.scene)

    if args.load_snapshot:
        scene_info = get_scene(args.load_snapshot)
        if scene_info is not None:
            args.load_snapshot = default_snapshot_filename(scene_info)  # noqa
        testbed.load_snapshot(args.load_snapshot)
    elif args.network:
        testbed.reload_network_from_file(args.network)

    if testbed.mode == ngp.TestbedMode.Sdf:
        testbed.tonemap_curve = ngp.TonemapCurve.ACES

    testbed.shall_train = args.train

    testbed.nerf.render_with_lens_distortion = True

    network_stem = (
        os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
    )
    if testbed.mode == ngp.TestbedMode.Sdf:
        setup_colored_sdf(testbed, args.scene)  # noqa

    old_training_step = 0
    n_steps = args.n_steps

    # If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
    # don't train by default and instead assume that the goal is to render screenshots,
    # compute PSNR, or render a video.
    if n_steps < 0 and (not args.load_snapshot or args.gui):
        n_steps = 35000

    tqdm_last_update = 0
    start = time.monotonic_ns()
    log_dict = {}
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="steps") as t:
            while testbed.frame():
                if testbed.want_repl():
                    repl(testbed)  # noqa

                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                # logging
                if not testbed.training_step % args.log_freq:
                    psnr = -10.0 * np.log(testbed.loss) / np.log(10.0)
                    elapsed_time = (time.monotonic_ns() - start) * 1e-9

                    if args.log_imgs and args.screenshot_dir:
                        # log images and training info
                        outname = os.path.join(
                            args.screenshot_dir, args.scene + "_" + network_stem
                        )
                        image = testbed.render(
                            args.width or 1920,
                            args.height or 1080,
                            args.screenshot_spp,
                            True,
                        )

                        assert (
                            image[:, :, :3].shape == groundtruth.shape
                        ), "image and groundtruth shapes not equal"
                        residual = (
                            groundtruth - unmultiply_alpha(image)[:, :, :3]  # noqa
                        )

                        res_mean = np.mean(residual)
                        res_mean_r = np.mean(residual[:, :, 0])
                        res_mean_g = np.mean(residual[:, :, 1])
                        res_mean_b = np.mean(residual[:, :, 2])

                        if os.path.dirname(outname) != "":
                            os.makedirs(os.path.dirname(outname), exist_ok=True)

                        # write_image(outname + ".png", image)
                        # image = linear_to_srgb(image)
                        # residual = linear_to_srgb(residual)

                        log_dict = {
                            "loss": np.log(testbed.loss) / np.log(10.0),
                            "training_step": testbed.training_step,
                            "psnr": psnr,
                            "elapsed_time": elapsed_time,
                            # "encoded_image": wandb.Image(image),
                            # "residual": wandb.Image(residual),
                            "res_mean": res_mean,
                            "res_mean_r": res_mean_r,
                            "res_mean_g": res_mean_g,
                            "res_mean_b": res_mean_b,
                        }
                    else:
                        # log training info only
                        log_dict = {
                            "loss": np.log(testbed.loss) / np.log(10.0),
                            "training_step": testbed.training_step,
                            "psnr": psnr,
                            "elapsed_time": elapsed_time,
                        }

                    # WandB logging
                    wandb.log(log_dict)

                # update tqdm
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    # Total training time
    elapsed_time = (time.monotonic_ns() - start) * 1e-9

    outfilename = ""
    if args.save_snapshot:
        outfilename = (
            os.path.splitext(os.path.split(args.scene)[1])[0]
            + "-"
            + timestamp
        )
        testbed.save_snapshot(args.save_snapshot + "/" + outfilename + ".ingp", False)

    config["name"] = args.name
    config["logging_images"] = args.log_imgs
    config["logging_frequency"] = args.log_freq
    config["n_params"] = testbed.n_params()
    config["n_enc_params"] = testbed.n_encoding_params()
    config["n_mlp_params"] = testbed.n_params() - testbed.n_encoding_params()
    config["snapshot_filename"] = outfilename
    wandb.config.update(config)

    image = None
    residual = None
    outname = None
    if args.screenshot_dir:
        outfilename = os.path.splitext(os.path.split(args.scene)[1])[0]
        outname = os.path.join(args.screenshot_dir, outfilename + "_" + network_stem)
        print(f"Rendering {outname}.png")
        image = testbed.render(
            args.width or 1920, args.height or 1080, args.screenshot_spp, True
        )
        if os.path.dirname(outname) != "":
            os.makedirs(os.path.dirname(outname), exist_ok=True)
        write_image(outname + "_image.png", image)  # noqa
        residual = groundtruth - unmultiply_alpha(image)[..., :3]  # noqa
        write_image(outname + "_residual.png", residual)  # noqa
        im_plus_residual = residual + unmultiply_alpha(image)[..., :3]  # noqa
        write_image(outname + "_residual+image.png", im_plus_residual)  # noqa

    res_mean = np.mean(residual)
    res_mean_r = np.mean(residual[:, :, 0])
    res_mean_g = np.mean(residual[:, :, 1])
    res_mean_b = np.mean(residual[:, :, 2])

    psnr = -10.0 * np.log(testbed.loss) / np.log(10.0)

    wandb.log(
        {
            "loss": np.log(testbed.loss) / np.log(10.0),
            "psnr": psnr,
            "training_step": testbed.training_step,
            "elapsed_time": elapsed_time,
            "encoded_image": wandb.Image(outname + "_image.png"),
            "residual": wandb.Image(outname + "_residual.png"),
            "residual_plus_image": wandb.Image(outname + "_residual+image.png"),
            "res_mean": res_mean,
            "res_mean_r": res_mean_r,
            "res_mean_g": res_mean_g,
            "res_mean_b": res_mean_b,
        }
    )

    run.finish()


if __name__ == "__main__":
    args = parse_args()

    path = "/cluster/home/jpostels/nnaimi/instant-ngp/configs/image/sweep-encoding.json"
    with open(path, "r") as f:
        sweep_config = json.load(f)

    # Define sweep config
    sweep_configuration = {
        "method": "grid",
        "name": "sweep-pca-study",
        "metric": {"goal": "maximize", "name": "psnr"},
        "parameters": sweep_config,
    }

    print("sweep configured.")
    # Initialize sweep by passing in config.
    # (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity="inrcompression",
        project="instantNGP-4-compression",
    )

    print("sweep instantiated.")

    # Begin the sweep
    wandb.agent(sweep_id, function=lambda: main(args), count=200)
    print("sweep conducted.")
