from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from enum import IntEnum
import viser
import nerfview
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.gaussians.basics import *

logger = logging.getLogger()

class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3

def lr_scheduler_fn(
    cfg: OmegaConf,
    lr_init: float
):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.
        
        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func

class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses
        print(losses)
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.step = 0
        self.device = device
        
        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images
        
        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb'
        ]
        self.gaussian_classes = {}
        self._init_models()
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False
        
        # init losses fn
        self._init_losses()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.step = 0

        # background color
        self.back_color = torch.zeros(3).to(self.device)
    
        # for evaluation
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices # will be override
        
        # a simple viewer for background visualization
        self.viewer = None
    
    
    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices
    
    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()
    
    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.res_schedule.downscale_times - self.step // self.res_schedule.double_steps), 0)
        else:
            return 1
        
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
        
    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")
    
    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")
    
    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())
                 
        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]
            
            raw_optim_cfg = class_optim_cfg.get(component_name, None)
            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
            if raw_optim_cfg.get("lr_final", None) is not None:
                sched_cfg = OmegaConf.create({
                    "opt_after": raw_optim_cfg.get('opt_after', 0),
                    "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                    "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                    "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                    "lr_final": raw_optim_cfg.get('lr_final', None),
                    "ramp": raw_optim_cfg.get('ramp', "cosine"),
                })
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
    
    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(pred, gt, reduction="mean")
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(pred, gt, limit=0.1, reduction="mean")
        self.sky_opacity_loss_fn = sky_opacity_loss_fn
        
        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss
            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )
        self.depth_loss_fn = depth_loss_fn
        
        if self.losses_dict.get("perception", None) is not None:
            self.perception_loss_fn = YOLOCIoUPerceptionLoss()
            
        if self.losses_dict.get("bbox", None) is not None:
            self.bbox_loss_fn = BBoxSSIMLoss()
    
    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()
    
    def optimizer_step(self) -> None:
        # for params_name, optimizer in self.optimizers.items():
        #     class_name = params_name.split("#")[0]
        #     component_name = params_name.split("#")[1]
        #     max_norm = self.model_config[class_name]["optim"][component_name].get("max_norm", None)
        #     if max_norm is not None:
        #         self.grad_scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(self.param_groups[params_name], max_norm)
        #     if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
        #         self.grad_scaler.step(optimizer)
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()
        
    def postprocess_per_train_step(self, step: int) -> None:
        radii = self.info["radii"]
        if self.render_cfg.absgrad:
            grads = self.info["means2d"].absgrad.clone()
        else:
            grads = self.info["means2d"].grad.clone()
        grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size
        grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size
        
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            
            self.models[class_name].postprocess_per_train_step(
                step=step,
                optimizer=self.optimizer,
                radii=radii[0, gaussian_mask],
                xys_grad=grads[0, gaussian_mask],
                last_size=max(self.info["width"], self.info["height"])
            )
        
        # viewer
        if self.viewer is not None:
            num_train_rays_per_step = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)
    
    def update_visibility_filter(self) -> None:
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            self.models[class_name].cur_radii = self.info["radii"][0, gaussian_mask]

    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)
        
        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"]
        )
        
        return camera_dict

    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor # leave it here for future use
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
        }
        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue
    
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=None        # to save some extra information (TODO) more flexible way
        )
        
        return gaussians
    
    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
    
        def render_fn(opaticy_mask=None, return_info=False):
            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=gs.rgbs,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
            rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], info
        
        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity
        }
        
        if self.training:
            self.info["means2d"].retain_grad()
        
        return results, render_fn

    def affine_transformation(
        self,
        rgb_blended: torch.Tensor,
        image_infos: Dict[str, torch.Tensor]
        ):
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]
            
            return rgb_transformed
        else:       
            return rgb_blended
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        return outputs
    
    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())
        self.grad_scaler.scale(total_loss).backward()
        self.optimizer_step()
        
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr
                
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}
        
        if "egocar_masks" in image_infos:
            # in the case of egocar, we need to mask out the egocar region
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
            
        gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
        predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]
        
        gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
        pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask
        
        # rgb loss
        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...], predicted_rgb.permute(2, 0, 1)[None, ...])
        loss_dict.update({
            "rgb_loss": self.losses_dict.rgb.w * Ll1,
            "ssim_loss": self.losses_dict.ssim.w * simloss,
        })
        
        # mask loss
        if self.sky_opacity_loss_fn is not None:
            sky_loss_opacity = self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask) * self.losses_dict.mask.w
            loss_dict.update({"sky_loss_opacity": sky_loss_opacity})
        
        # depth loss
        if self.depth_loss_fn is not None:
            gt_depth = image_infos["lidar_depth_map"] 
            lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
            pred_depth = outputs["depth"]
            depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)
            
            lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
            if lidar_w_decay > 0:
                decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
            else:
                decay_weight = 1
            depth_loss = depth_loss * self.losses_dict.depth.w * decay_weight
            loss_dict.update({"depth_loss": depth_loss})
            
        # ----- reg loss -----
        opacity_entropy_reg = self.losses_dict.get("opacity_entropy", None)
        if opacity_entropy_reg is not None:
            pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
            loss_dict.update({
                "opacity_entropy_loss": opacity_entropy_reg.w * (-pred_opacity * torch.log(pred_opacity)).mean()
            })
            
        # from pvg: https://github.com/fudan-zvg/PVG/blob/b4162a9135282e0f3c929054f16be1b3fbacd77a/train.py#L161
        inverse_depth_smoothness_reg = self.losses_dict.get("inverse_depth_smoothness", None)
        if inverse_depth_smoothness_reg is not None:
            inverse_depth = 1 / (outputs["depth"] + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                image_infos["pixels"][None].permute(0, 3, 1, 2)
            )
            loss_dict.update({
                "inverse_depth_smoothness_loss": inverse_depth_smoothness_reg.w * loss_inv_depth
            })
            
        # affine reg loss
        affine_reg = self.losses_dict.get("affine", None)
        if affine_reg is not None and "Affine" in self.models:
            affine_trs = self.models['Affine']({"img_idx": image_infos["img_idx"].flatten()[0]})
            reg_mat = torch.eye(3, device=self.device)
            reg_shift = torch.zeros(3, device=self.device)
            loss_affine = torch.abs(affine_trs[..., :3, :3] - reg_mat).mean() + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
            loss_dict.update({
                "affine_loss": affine_reg.w * loss_affine
            })

        # dynamic region loss
        dynamic_region_weighted_losses = self.losses_dict.get("dynamic_region", None)
        if dynamic_region_weighted_losses is not None:
            weight_factor = dynamic_region_weighted_losses.get("w", 1.0)
            start_from = dynamic_region_weighted_losses.get("start_from", 0)
            if self.step == start_from:
                self.render_dynamic_mask = True
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                
                if dynamic_pred_mask.sum() > 0:
                    Ll1 = torch.abs(gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]).mean()
                    loss_dict.update({
                        "vehicle_region_rgb_loss": weight_factor * Ll1,
                    })
        perception_losses = self.losses_dict.get("perception", None)
        if perception_losses is not None:
            rec_image = predicted_rgb
            gt_image = gt_rgb
            target_h, target_w = 320, 480

            # [H,W,3] -> [1,3,H,W]
            rec_image = rec_image.permute(2, 0, 1).unsqueeze(0)
            gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)

            # resize
            rec_image = F.interpolate(rec_image, size=(target_h, target_w), mode='bilinear', align_corners=False)
            gt_image = F.interpolate(gt_image, size=(target_h, target_w), mode='bilinear', align_corners=False)

            perception_loss = self.perception_loss_fn.get_perception_loss(rec_image, gt_image) * perception_losses.get("w", 1.0)
            loss_dict.update({"perception_loss": perception_loss})
        
        bbox_losses = self.losses_dict.get("bbox", None)
        if bbox_losses is not None:
            rec_image = predicted_rgb
            gt_image = gt_rgb
            rec_image = rec_image.permute(2, 0, 1).unsqueeze(0)
            gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)
            
            bbox_loss = self.bbox_loss_fn.get_bbox_ssim_loss(rec_image, gt_image) * bbox_losses.get("w", 1.0)
            loss_dict.update({"bbox_loss": bbox_loss})
            
        # compute gaussian reg loss
        for class_name in self.gaussian_classes.keys():
            class_reg_loss = self.models[class_name].compute_reg_loss()
            for k, v in class_reg_loss.items():
                loss_dict[f"{class_name}_{k}"] = v
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict
    
    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict
    
    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update({
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "step": self.step,
        })
        if not only_model:
            state_dict.update({
                "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
                # "lr_schedulers": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
                # "grad_scaler": self.grad_scaler.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model
        model_state_dict = state_dict.pop("models")
        for class_name in self.models.keys():
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                logger.warning(f"Cannot find {class_name} in the checkpoint")
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")
        
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=True)
        
    def save_checkpoint(
        self,
        log_dir: str,
        save_only_model: bool=True,
        is_final: bool=False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")
        
    def init_viewer(self, port: int = 8080):
        # a simple viewer for background ONLY visualization
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            Ks=K,
            H=H,
            W=W
        )
        
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }
        for class_name in ["Background"]:
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue

            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None
        )
        
        render_colors, _, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        return render_colors[0].cpu().numpy()
    
def denormalize_yolo_bboxes(gt_boxes, H, W):
    if len(gt_boxes) == 0:
        return np.zeros((0,4), dtype=np.float32)

    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    x_center = gt_boxes[:,0] * W
    y_center = gt_boxes[:,1] * H
    w = gt_boxes[:,2] * W
    h = gt_boxes[:,3] * H

    x1 = x_center - w/2
    y1 = y_center - h/2
    x2 = x_center + w/2
    y2 = y_center + h/2

    return np.stack([x1, y1, x2, y2], axis=1)

def _input_gt_bbox(source_path):
    gt_bboxes_path = Path(source_path).joinpath("bboxes", "labels")
    gt_bboxes = {}
    for txt_file in gt_bboxes_path.glob("*.txt"):
        gt_bboxes[txt_file.stem] = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    gt_bboxes[txt_file.stem].append([x1, y1, x2, y2])
    return gt_bboxes

from ultralytics import YOLO

class YOLOCIoUPerceptionLoss:
    def __init__(self, source_path, model_path='yolov8s.pt', conf_threshold=0.5, device='cuda', lambda_perception=0.001):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.lambda_perception = lambda_perception
        self.gt_bboxes = _input_gt_bbox(source_path)
        

    def _run_model(self, img):
        img_clamped = torch.clamp(img, 0.0, 1.0)
        with torch.no_grad():
            results = self.model(img_clamped, conf=self.conf_threshold, verbose=False)
        boxes_list, scores_list = [], []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # [N,4]
            scores = r.boxes.conf.cpu().numpy()  # [N]
            mask = scores >= self.conf_threshold
            boxes_list.append(boxes[mask])
            scores_list.append(scores[mask])
        if len(boxes_list) == 0 or boxes_list[0].size == 0:
            return np.zeros((0,4)), np.zeros((0,))
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        return boxes, scores

    @staticmethod
    def ciou_loss(pred_boxes, gt_boxes, eps=1e-7):
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return torch.tensor(0.0, device='cuda')

        pred = torch.tensor(pred_boxes, dtype=torch.float32, device='cuda')
        gt = torch.tensor(gt_boxes, dtype=torch.float32, device='cuda')

        pred_w = pred[:,2]-pred[:,0]
        pred_h = pred[:,3]-pred[:,1]
        gt_w = gt[:,2]-gt[:,0]
        gt_h = gt[:,3]-gt[:,1]

        pred_cx = (pred[:,0]+pred[:,2])/2
        pred_cy = (pred[:,1]+pred[:,3])/2
        gt_cx = (gt[:,0]+gt[:,2])/2
        gt_cy = (gt[:,1]+gt[:,3])/2

        inter_x1 = torch.max(pred[:,0], gt[:,0])
        inter_y1 = torch.max(pred[:,1], gt[:,1])
        inter_x2 = torch.min(pred[:,2], gt[:,2])
        inter_y2 = torch.min(pred[:,3], gt[:,3])
        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        inter_area = inter_w * inter_h

        union_area = pred_w*pred_h + gt_w*gt_h - inter_area
        iou = inter_area / (union_area + eps)

        c_x1 = torch.min(pred[:,0], gt[:,0])
        c_y1 = torch.min(pred[:,1], gt[:,1])
        c_x2 = torch.max(pred[:,2], gt[:,2])
        c_y2 = torch.max(pred[:,3], gt[:,3])
        c_diag = (c_x2-c_x1)**2 + (c_y2-c_y1)**2

        rho2 = (pred_cx-gt_cx)**2 + (pred_cy-gt_cy)**2

        v = (4 / (np.pi**2)) * torch.pow(torch.atan(gt_w/(gt_h+eps)) - torch.atan(pred_w/(pred_h+eps)), 2)
        alpha = v / (1 - iou + v + eps)

        ciou = iou - rho2/(c_diag+eps) - alpha*v
        return (1 - ciou).mean()

    def get_perception_loss(self, rec_image, frame_name):
        rec_boxes, rec_scores = self._run_model(rec_image)
        gt_boxes = self.gt_bboxes.get(frame_name, [])
        H, W = rec_image.shape[-2:]
        gt_boxes = denormalize_yolo_bboxes(gt_boxes, H, W)

        max_boxes = min(len(rec_boxes), len(gt_boxes), 10)
        if max_boxes == 0:
            return torch.tensor(0.0, device=self.device)

        rec_boxes = rec_boxes[:max_boxes]
        gt_boxes = gt_boxes[:max_boxes]

        loss = self.ciou_loss(rec_boxes, gt_boxes)
        return self.lambda_perception * loss
    

class BBoxSSIMLoss:
    def __init__(self, source_path, window_size=9, conf_thres=0.5):
        self.window_size = window_size
        self.conf_thres = conf_thres
        self.gt_bboxes = _input_gt_bbox(source_path)

    def get_bbox_ssim_loss(self, re_img, gt_img, frame_name):
        """
        re_img, gt_img: (1, C, H, W), range [0,1]
        return: scalar loss
        """


        gt_boxes = self.gt_bboxes.get(frame_name, [])
        H, W = re_img.shape[-2:]
        gt_boxes = denormalize_yolo_bboxes(gt_boxes, H, W)

        # bboxes = []
        # for r in gt_boxes:
        #     for box in r.boxes.xyxy:  # [x1, y1, x2, y2]
        #         bboxes.append(box.tolist())

        if len(gt_boxes) == 0:
            return (re_img * 0).sum()

        losses = []
        for bbox in gt_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            crop1 = re_img[..., y1:y2, x1:x2]
            crop2 = gt_img[..., y1:y2, x1:x2]

            if crop2.size(-1) < self.window_size or crop2.size(-2) < self.window_size:
                continue

            ssim_val = ssim(crop1, crop2, window_size=self.window_size, size_average=True)
            losses.append(1 - ssim_val)

        if len(losses) == 0:
            return (re_img * 0).sum()

        return torch.stack(losses).mean()
