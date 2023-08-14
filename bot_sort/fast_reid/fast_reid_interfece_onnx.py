import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import onnxruntime
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch




def normalize_to_unit_length(arr):
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    normalized_arr = arr / norm
    return normalized_arr


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    #cfg.MODEL.DEVICE='cpu'
    cfg.freeze()

    return cfg



def postprocess(features):
    features = normalize_to_unit_length(features)
    features = features
    return features

class FastReIDInterface:
    def __init__(self, config_file, weights_path, device,debug, batch_size=8):
        super(FastReIDInterface, self).__init__()
        if str(device) != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # self.device = device
        # print("The device in fast reid is",self.device)
        self.batch_size = batch_size
       
        self.debug = debug
        # Load ONNX model with ONNX Runtime
        # providers = ["TensorrtExecutionProvider"]
        # if str(device) == "cpu":
        #     providers.append("CPUExecutionProvider")
            
        # else:
        #     providers.append("CUDAExecutionProvider")
        # providers = ["TensorrtExecutionProvider","CUDAExecutionProvider"]

        if self.debug:
            print("ONNX Runtime device for fastreid",onnxruntime.get_device())
            # print("Providers",providers)

        self.session = onnxruntime.InferenceSession(weights_path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])#,provider_options=[{"device_id":1}])
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path, "MODEL.DEVICE", self.device])
        print("ONNX Runtime device for fastreid",onnxruntime.get_device())
        # print(onnxruntime.get_available_providers())
        # print(onnxruntime.get_all_providers())
        
    def inference(self, image, detections):
        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
            patch = patch[:, :, ::-1]
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            if self.device != 'cpu':
                patch = patch.to(device=self.device).half()
                patch = patch.float()       # Changed this for GPU
            else:
                patch = patch.to(device=self.device)

            patches.append(patch)
            # patch = torch.as_tensor(patch.astype('float32'))

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))

        for patches in batch_patches:
            patches_ = torch.clone(patches)
            
            # Run model using ONNX Runtime
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            input_data = patches_.cpu().numpy()
            if self.debug:
                print("input shape in the Fast_reid_model",input_data.shape)
            

            pred = self.session.run([output_name], {input_name: input_data})[0]
            pred[np.isinf(pred)] = 1.0

            feat = postprocess(pred)

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()
                        print(patch_np.shape)
                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            features = np.vstack((features, feat))

        return features
