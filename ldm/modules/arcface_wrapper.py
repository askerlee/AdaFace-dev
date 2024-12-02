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
    def __init__(self, device='cpu', dtype=torch.float16, ckpt_path='models/arcface-resnet18_110.pth'):
        super(ArcFaceWrapper, self).__init__()
        self.arcface = resnet_face18(False)
        ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')
        for key in list(ckpt_state_dict.keys()):
            new_key = key.replace("module.", "")
            ckpt_state_dict[new_key] = ckpt_state_dict.pop(key)

        self.arcface.load_state_dict(ckpt_state_dict)
        self.arcface.eval()
        self.dtype = dtype
        self.arcface.to(device, dtype=self.dtype)

        self.retinaface = RetinaFaceClient(device=device)
        # We keep retinaface at float32, as it doesn't require grad and won't consume much memory.
        self.retinaface.eval()

        for param in self.arcface.parameters():
            param.requires_grad = False
        for param in self.retinaface.parameters():
            param.requires_grad = False

    # Suppose images_ts has been normalized to [-1, 1].
    # Cannot wrap this function with @torch.compile. Otherwise a lot of warnings will be spit out.
    def embed_image_tensor(self, images_ts, T=20, use_whole_image_if_no_face=False, enable_grad=True):
        # retina_crop_face() crops on the input tensor, so that computation graph is preserved.
        # The cropping param computation is wrapped with torch.no_grad().
        faces, failed_indices = self.retinaface.crop_faces(images_ts, out_size=(128, 128), T=T, 
                                                           use_whole_image_if_no_face=use_whole_image_if_no_face)
        
        # No face detected in any instances in the batch.
        if faces is None:
            return None, failed_indices
        # Arcface takes grayscale images as input
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], device=images_ts.device).view(1, 3, 1, 1)
        # Convert RGB to grayscale
        faces_gray = (faces * rgb_to_gray_weights).sum(dim=1, keepdim=True)
        # Resize to (128, 128)
        faces_gray = F.interpolate(faces_gray, size=(128, 128), mode='bilinear', align_corners=False)
        with torch.set_grad_enabled(enable_grad):
            faces_emb = self.arcface(faces_gray.to(self.dtype))
        return faces_emb, failed_indices

    # T: minimal face height/width to be detected.
    # ref_images: the groundtruth images.
    # aligned_images: the generated   images.
    def calc_arcface_align_loss(self, ref_images, aligned_images, T=20, use_whole_image_if_no_face=False):
        embs1, failed_indices1 = \
            self.embed_image_tensor(ref_images, T, use_whole_image_if_no_face=False, enable_grad=False)
        embs2, failed_indices2 = \
            self.embed_image_tensor(aligned_images, T, use_whole_image_if_no_face=use_whole_image_if_no_face, 
                                    enable_grad=True)
        
        if len(failed_indices1) > 0:
            print(f"Failed to detect faces in ref_images-{failed_indices1}")
            return torch.tensor(0.0, device=ref_images.device)
        if len(failed_indices2) > 0:
            print(f"Failed to detect faces in aligned_images-{failed_indices2}")
            return torch.tensor(0.0, device=ref_images.device)

        # Repeat groundtruth embeddings to match the number of generated embeddings.
        if len(embs1) < len(embs2):
            embs1 = embs1.repeat(len(embs2)//len(embs1), 1)
            
        arcface_align_loss = F.cosine_embedding_loss(embs1, embs2, torch.ones(embs1.shape[0]).to(embs1.device))
        print(f"Arcface align loss: {arcface_align_loss.item():.2f}")
        return arcface_align_loss
