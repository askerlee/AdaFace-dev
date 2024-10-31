import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from evaluation.arcface_resnet import resnet_face18
from evaluation.retinaface_pytorch import RetinaFaceClient

def load_image_for_arcface(img_path, device='cpu'):
    # cv2.imread ignores the alpha channel by default.
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]    
    image = image.astype(np.float32, copy=False)
    # Normalize to [-1, 1].
    image -= 127.5
    image /= 127.5
    image_ts = torch.from_numpy(image).to(device)
    return image_ts

class ArcFaceWrapper(nn.Module):
    def __init__(self, device='cpu', ckpt_path='models/arcface-resnet18_110.pth'):
        super(ArcFaceWrapper, self).__init__()
        self.arcface = resnet_face18(False)
        ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')
        for key in list(ckpt_state_dict.keys()):
            new_key = key.replace("module.", "")
            ckpt_state_dict[new_key] = ckpt_state_dict.pop(key)

        self.arcface.load_state_dict(ckpt_state_dict)
        self.arcface.eval()
        self.arcface.to(device)

        self.retinaface = RetinaFaceClient(device=device)
        for param in self.retinaface.parameters():
            param.requires_grad = False
        self.retinaface.eval()
        self.retinaface.to(device)

    # Suppose images_ts has been normalized to [-1, 1].
    def embed_tensor(self, images_ts):
        # retina_crop_face() crops on the input tensor, so that computation graph is preserved.
        faces, failed_indices = self.retinaface.crop_faces(images_ts, out_size=(128, 128))
        # No face detected in any instances in the batch.
        if faces is None:
            return None, failed_indices
        # Arcface takes grayscale images as input
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], device=images_ts.device).view(1, 3, 1, 1)
        # Convert RGB to grayscale
        faces_gray = (faces * rgb_to_gray_weights).sum(dim=1, keepdim=True)
        # Resize to (128, 128)
        faces_gray = F.interpolate(faces_gray, size=(128, 128), mode='bilinear', align_corners=False)
        faces_emb = self.arcface(faces_gray)
        return faces_emb, failed_indices

    def calc_arcface_align_loss(self, images1, images2):
        embs1, failed_indices1 = self.embed_tensor(images1)
        embs2, failed_indices2 = self.embed_tensor(images2)
        if len(failed_indices1) > 0:
            print(f"Failed to detect faces in images1-{failed_indices1}")
            return torch.tensor(0.0, device=images1.device)
        if len(failed_indices2) > 0:
            print(f"Failed to detect faces in images2-{failed_indices2}")
            return torch.tensor(0.0, device=images1.device)
                                
        arcface_align_loss = F.cosine_embedding_loss(embs1, embs2, torch.ones(embs1.shape[0]).to(embs1.device))
        return arcface_align_loss
