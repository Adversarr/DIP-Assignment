import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        ### FILL: Compute the Jacobian of the projection
        # Extract intrinsic parameters
        fx = K[0, 0]
        fy = K[1, 1]

        # Extract camera-space coordinates
        x = cam_points[:, 0]  # (N,)
        y = cam_points[:, 1]  # (N,)
        z = cam_points[:, 2]  # (N,)

        # Avoid division by zero (though z > 0 due to clipping in the earlier step)
        eps = 1e-6
        z = z.clamp(min=eps)

        # Compute Jacobian for each point
        J_proj[:, 0, 0] = fx / z                   # ∂u/∂x
        J_proj[:, 0, 1] = 0                        # ∂u/∂y
        J_proj[:, 0, 2] = -(fx * x) / (z * z)      # ∂u/∂z

        J_proj[:, 1, 0] = 0                        # ∂v/∂x
        J_proj[:, 1, 1] = fy / z                   # ∂v/∂y
        J_proj[:, 1, 2] = -(fy * y) / (z * z)      # ∂v/∂z
        
        # Transform covariance to camera space
        ### FILL: Aplly world to camera rotation to the 3d covariance matrix
        covs_cam = torch.matmul(R, torch.matmul(covs3d, R.T))  # (N, 3, 3)
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization
        ### FILL: compute the gaussian values
        # Compute determinant for normalization (N,)
        det_cov = covs2D[:, 0, 0] * covs2D[:, 1, 1] - covs2D[:, 0, 1] * covs2D[:, 1, 0]  # (N,)
        norm_factor = 1.0 / (2.0 * torch.pi * torch.sqrt(det_cov))  # (N,)

        # Compute inverse covariance matrix (N, 2, 2)
        inv_covs = torch.inverse(covs2D)  # (N, 2, 2)
        
        # Compute Mahalanobis distance: dx^T @ inv_cov @ dx (N, H, W)
        dx = dx.reshape(N, H * W, 2)  # (N, H*W, 2)
        mahalanobis_dist = torch.einsum('nhi,nij,nhj->nh', dx, inv_covs, dx)  # (N, H*W)
        mahalanobis_dist = mahalanobis_dist.reshape(N, H, W)  # (N, H, W)
        
        # Compute Gaussian values (N, H, W)
        gaussian = norm_factor[:, None, None] * torch.exp(-0.5 * mahalanobis_dist)  # (N, H, W)
        
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        ### FILL:
        ### weights = ... # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
