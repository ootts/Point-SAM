from point_sam.build_model import build_point_sam
import numpy as np
import torch

# load model
from point_sam.model import PointCloudEncoder

ckpt_path = "model.safetensors"
model = build_point_sam(ckpt_path, 512, 64)  # (ckpt_path, num_centers, KNN size)
pc_encoder: PointCloudEncoder = model.pc_encoder.cuda()

xyz = np.random.rand(1, 1000, 3)  # (batch_size, num_points, 3)
xyz = torch.tensor(xyz).float().cuda()
rgb = np.random.rand(1, 1000, 3)  # (batch_size, num_points, 3)
rgb = torch.tensor(rgb).float().cuda()

# encode point cloud
"""
embedding is the point cloud's embedded features. [B, num_centers, 1024]
"""
for i in range(100):
    embeddings, patches = pc_encoder(xyz, rgb)

# other useful information
centers = patches["centers"]  # centers sampled by FPS. [B, num_patches, 3]
knn_idx = patches["knn_idx"]  # K nearest "CENTER"s for each "POINT", this feature is used for interpolation. [B, N, K]

print()
