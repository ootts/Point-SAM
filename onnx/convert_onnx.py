import os
import torch
import numpy as np
import torch.nn as nn

from point_sam.model import PointCloudEncoder
from point_sam.build_model import build_point_sam


class PointSAMEncoderOnnx(nn.Module):
    def __init__(self, pc_encoder):
        super(PointSAMEncoderOnnx, self).__init__()
        self.pc_encoder = pc_encoder

    def forward(self, patch_features, centers):
        """
        :param patch_features: bsz,num_patches,patch_size,6
        :param centers: bsz,num_patches,3
        :return:
        """
        return self.pc_encoder.forward_onnx(patch_features, centers)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='')
    parser.add_argument('--output', default='onnx')
    parser.add_argument('--num_centers', default=512,type=int)
    args = parser.parse_args()
    ckpt_path = args.input
    num_centers = args.num_centers
    model = build_point_sam(ckpt_path, num_centers, 64)  # (ckpt_path, num_centers, KNN size)
    pc_encoder: PointCloudEncoder = model.pc_encoder.cuda()

    model = PointSAMEncoderOnnx(pc_encoder)

    patch_features = np.random.rand(1, num_centers, 64, 6)
    patch_features = torch.tensor(patch_features).float().cuda()
    centers = np.random.rand(1, num_centers, 3)
    centers = torch.tensor(centers).float().cuda()
    # patch_features, centers
    # Export torch model to ONNX
    output_onnx = args.output
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, (patch_features, centers), output_onnx,
                      export_params=True,
                      verbose=False,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["patch_features", "centers"],
                      output_names=["output"],
                      dynamic_axes={"patch_features": {0: "batch"},
                                    "centers": {0: "batch"},
                                    "output": {0: "batch"}
                                    })


if __name__ == '__main__':
    main()
