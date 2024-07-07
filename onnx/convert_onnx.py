import os
import torch
import numpy as np
import torch.nn as nn
import os.path as osp

from point_sam.model import PointCloudEncoder
from point_sam.build_model import build_point_sam


class PointSAMEncoderOnnx(nn.Module):
    def __init__(self, pc_encoder):
        super(PointSAMEncoderOnnx, self).__init__()
        self.pc_encoder = pc_encoder

    def forward(self, xyz, rgb):
        """
        :param xyz: bsz,num_points,3
        :param rgb: bsz,num_points,3
        :return:
        """
        return self.pc_encoder(xyz, rgb)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='onnx')
    args = parser.parse_args()
    ckpt_path = "model.safetensors"
    model = build_point_sam(ckpt_path, 512, 64)  # (ckpt_path, num_centers, KNN size)
    pc_encoder: PointCloudEncoder = model.pc_encoder.cuda()

    model = PointSAMEncoderOnnx(pc_encoder)

    # left_tensor = torch.rand(20, 3, 112, 112).float().cuda()
    # right_tensor = torch.rand(20, 3, 112, 112).float().cuda()
    # left_tensor, right_tensor = torch.load('tmp/left_right_roi_images.pth', 'cuda')
    xyz = np.random.rand(1, 512, 3)  # (batch_size, num_points, 3)
    xyz = torch.tensor(xyz).float().cuda()
    rgb = np.random.rand(1, 512, 3)  # (batch_size, num_points, 3)
    rgb = torch.tensor(rgb).float().cuda()

    # Export torch model to ONNX
    output_onnx = args.output
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, (xyz, rgb), output_onnx,
                      export_params=True,
                      verbose=False,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["xyz", "rgb"],
                      output_names=["output"],
                      dynamic_axes={"xyz": {0: "batch"},
                                    "rgb": {0: "batch"},
                                    "output": {0: "batch"}
                                    })

    # simp_onnx = output_onnx.replace('.onnx', '-simp.onnx')
    # onnxsim_path = sys.executable.replace("/bin/python", "/bin/onnxsim")
    # os.system(f"{onnxsim_path} {output_onnx} {simp_onnx}")
    #
    # print('to engine')
    # engine_path = osp.join(cfg.trt.convert_to_trt.output_path, "idispnet.engine")
    # trtexec_path = osp.expanduser(cfg.trt.convert_to_trt.trtexec_path)
    # cmd = f"{trtexec_path} --onnx={simp_onnx}"
    # if cfg.trt.convert_to_trt.fp16:
    #     cmd = cmd + " --fp16"
    #     engine_path = engine_path.replace(".engine", "-fp16.engine")
    # cmd = cmd + " --minShapes=left_input:1x3x112x112,right_input:1x3x112x112" \
    #             " --optShapes=left_input:4x3x112x112,right_input:4x3x112x112" \
    #             " --maxShapes=left_input:20x3x112x112,right_input:20x3x112x112"
    # cmd = cmd + f" --workspace=40960 --saveEngine={engine_path}  --tacticSources=-cublasLt,+cublas"
    # os.system(cmd)


if __name__ == '__main__':
    main()
