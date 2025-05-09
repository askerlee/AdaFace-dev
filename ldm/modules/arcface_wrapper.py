import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from evaluation.arcface_resnet import resnet_face18
from evaluation.retinaface_pytorch import RetinaFaceClient

class MaskedGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, mask, debug=False):
        ctx.save_for_backward(mask, debug)
        output = input_
        if debug:
            print(f"input: {input_.abs().mean().detach().item()}")
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        mask, debug = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_output2 = grad_output * mask
            if debug:
                print(f"grad_output2: {grad_output2.abs().mean().detach().item()}")
        else:
            grad_output2 = None
        return grad_output2, None, None
    
class MaskedGradLayer(nn.Module):
    def __init__(self, mask=None, debug=False, *args, **kwargs):
        """
        A masked gradient layer.
        This layer has no parameters, and simply masks the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._mask = mask
        self._debug = torch.tensor(debug, requires_grad=False)

    def forward(self, input_):
        _debug = self._debug if hasattr(self, '_debug') else False
        return MaskedGrad.apply(input_, self._mask.to(input_.device), _debug)
    
def gen_masked_grad_layer(mask=None, debug=False):
    if mask is None:
        return nn.Identity()
    return MaskedGradLayer(mask, debug=debug)

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
    def embed_image_tensor(self, images_ts, T=20, embed_bg_faces=True,
                           enable_grad=True, fg_faces_grad_mask_ratios=(1, 0.3)):
        # retina_crop_face() crops on the input tensor, so that computation graph w.r.t. 
        # the input tensor is preserved.
        # But the cropping operation is wrapped with torch.no_grad().
        # fg_face_bboxes: long tensor of [BS, 4].
        # face_detected_inst_mask: binary tensor of [BS].
        fg_face_crops, bg_face_crops_flat, fg_face_bboxes, face_confidences, face_detected_inst_mask = \
            self.retinaface.crop_faces(images_ts, out_size=(128, 128), T=T)
        
        # No face detected in any instances in the batch. 
        # fg_face_bboxes is a tensor of the full image size, 
        # and doesn't indicate the face locations.
        if face_detected_inst_mask.sum() == 0:
            return None, None, None, None, face_confidences, face_detected_inst_mask
        
        # Arcface takes grayscale images as input
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], device=images_ts.device).view(1, 3, 1, 1)
        # Convert RGB to grayscale
        fg_faces_gray = (fg_face_crops * rgb_to_gray_weights).sum(dim=1, keepdim=True)
        fg_faces_gray = fg_faces_gray.to(self.dtype)
        # fg_faces_gray2 = rgb_to_grayscale(fg_face_crops)

        # Resize to (128, 128); arcface takes 128x128 images as input.
        fg_faces_gray = F.interpolate(fg_faces_gray, size=(128, 128), mode='bilinear', align_corners=False)
        fg_central_mask = torch.zeros_like(fg_faces_gray, device=fg_faces_gray.device)
        fg_border_mask  = torch.ones_like(fg_faces_gray, device=fg_faces_gray.device)
        fg_faces_grad_central_mask_ratio, fg_faces_grad_border_mask_ratio = fg_faces_grad_mask_ratios

        if 0 < fg_faces_grad_central_mask_ratio < 1:
            ML = int(fg_faces_gray.shape[3] * (1 - fg_faces_grad_central_mask_ratio) / 2)
            MR = fg_faces_gray.shape[3] - ML
            MT = int(fg_faces_gray.shape[2] * (1 - fg_faces_grad_central_mask_ratio) / 2)
            MB = fg_faces_gray.shape[2] - MT
            # fg_central_mask: The central fg_faces_grad_central_mask_ratio part of the face is filled with 1s.
            # This will stop the gradient from flowing through the border part of the face,
            # to avoid the face being encouraged to grow too big.
            fg_central_mask[:, :, MT:MB, ML:MR] = 1
            fg_central_masked_grad_layer = gen_masked_grad_layer(fg_central_mask, debug=False)
            fg_faces_gray_center = fg_central_masked_grad_layer(fg_faces_gray)
        else:
            fg_faces_gray_center = fg_faces_gray
            
        if 0 < fg_faces_grad_border_mask_ratio < 1:
            ML = int(fg_faces_gray.shape[3] * (1 - fg_faces_grad_border_mask_ratio) / 2)
            MR = fg_faces_gray.shape[3] - ML
            MT = int(fg_faces_gray.shape[2] * (1 - fg_faces_grad_border_mask_ratio) / 2)
            MB = fg_faces_gray.shape[2] - MT
            # fg_border_mask: The central fg_faces_grad_border_mask_ratio part of the face is filled with 0s.
            # This will stop the gradient from flowing through the central part of the face, 
            # to avoid totally destroying the identity captured by adaface.
            fg_border_mask[:, :, MT:MB, ML:MR] = 0
            fg_border_masked_grad_layer = gen_masked_grad_layer(fg_border_mask, debug=False)
            fg_faces_gray_border = fg_border_masked_grad_layer(fg_faces_gray)
        else:
            fg_faces_gray_border = None

        with torch.set_grad_enabled(enable_grad):
            # If some instances have no face detected, we still compute their face embeddings,
            # but such face embeddings shouldn't be used for loss computation, instead,
            # they should be filtered out by face_detected_inst_mask.
            fg_faces_emb_center = self.arcface(fg_faces_gray_center)
            if fg_faces_gray_border is not None:
                fg_faces_emb_border = self.arcface(fg_faces_gray_border)
            else:
                fg_faces_emb_border = fg_faces_emb_center    

        if embed_bg_faces and bg_face_crops_flat is not None:
            bg_faces_gray = (bg_face_crops_flat * rgb_to_gray_weights).sum(dim=1, keepdim=True)
            bg_faces_gray = F.interpolate(bg_faces_gray, size=(128, 128), mode='bilinear', align_corners=False)
            bg_faces_gray = bg_faces_gray.to(self.dtype)
            with torch.set_grad_enabled(enable_grad):
                bg_faces_emb = self.arcface(bg_faces_gray)
        else:
            bg_faces_emb = None

        return fg_faces_emb_center, fg_faces_emb_border, bg_faces_emb, fg_face_bboxes, \
               face_confidences, face_detected_inst_mask

    # T: minimal face height/width to be detected.
    # ref_images:     the groundtruth images, roughly normalized to [-1, 1] (could go beyond).
    # aligned_images: the generated   images, roughly normalized to [-1, 1] (could go beyond).
    def calc_arcface_align_loss(self, ref_images, aligned_images, T=20, 
                                fg_faces_grad_mask_ratios=(1, 0.3)):
        # ref_fg_face_bboxes: long tensor of [BS, 4], where BS is the batch size.
        ref_fg_faces_emb, _, _, ref_fg_face_bboxes, ref_fg_face_confidences, ref_face_detected_inst_mask = \
            self.embed_image_tensor(ref_images, T, embed_bg_faces=False,
                                    enable_grad=False, fg_faces_grad_mask_ratios=(-1, -1))
        # bg_embs are not separated by instances, but flattened. 
        # We don't align them, just suppress them. So we don't need the batch dimension.
        aligned_fg_faces_emb_center, aligned_fg_faces_emb_border, aligned_bg_faces_emb, \
          aligned_fg_face_bboxes, aligned_fg_face_confidences, aligned_face_detected_inst_mask = \
            self.embed_image_tensor(aligned_images, T, embed_bg_faces=True, enable_grad=True,
                                    fg_faces_grad_mask_ratios=fg_faces_grad_mask_ratios)
        
        zero_losses = [ torch.tensor(0., dtype=ref_images.dtype, device=ref_images.device) for _ in range(3) ]
        # As long as there's one instance in the reference batch that has no face detected, 
        # we don't compute the loss and return zero losses.
        # But only if all instances in the aligned batch have no face detected, we return zero losses.
        if (1 - ref_face_detected_inst_mask).sum() > 0:
            print(f"Failed to detect faces in some ref_images. Cannot compute arcface align loss")
            return zero_losses[0], zero_losses[1], zero_losses[2], None, \
                   aligned_fg_face_confidences, aligned_face_detected_inst_mask
        if aligned_face_detected_inst_mask.sum() == 0:
            print(f"Failed to detect faces in any aligned_images. Cannot compute arcface align loss")
            return zero_losses[0], zero_losses[1], zero_losses[2], None, \
                   aligned_fg_face_confidences, aligned_face_detected_inst_mask

        # If the numbers of instances in ref_fg_faces_emb and aligned_fg_faces_emb are different, then there's only one ref image, 
        # and multiple aligned images of the same person.
        # We repeat groundtruth embeddings to match the number of generated embeddings.
        if len(ref_fg_faces_emb) < len(aligned_fg_faces_emb_center):
            ref_fg_faces_emb = ref_fg_faces_emb.repeat(len(aligned_fg_faces_emb_center)//len(ref_fg_faces_emb), 1)
        
        # labels = 1: align the embeddings of the same person.
        # losses_arcface_align: tensor of [BS].
        # losses_arcface_align uses a center-filled grad mask to 
        # direct the optimization to focus on the central part of the face and ignore the border part,
        # so that the face will not grow too big.
        losses_arcface_align = F.cosine_embedding_loss(ref_fg_faces_emb, aligned_fg_faces_emb_center, 
                                                       torch.ones(ref_fg_faces_emb.shape[0]).to(ref_fg_faces_emb.device),
                                                       reduction='none')
        # Mask out the losses of instances that have no face detected.
        # aligned_face_detected_inst_mask: binary tensor of [BS].
        loss_arcface_align = (losses_arcface_align * aligned_face_detected_inst_mask).sum() / aligned_face_detected_inst_mask.sum()
        print(f"loss_arcface_align: {loss_arcface_align.detach().item():.2f}")
        # losses_fg_faces_suppress uses a border-filled grad mask to
        # direct the optimization to suppress the border part of the face 
        # so that the face will shrink from the border.
        losses_fg_faces_suppress = (aligned_fg_faces_emb_border ** 2).mean(dim=1)
        loss_fg_faces_suppress = (losses_fg_faces_suppress * aligned_face_detected_inst_mask).sum() / aligned_face_detected_inst_mask.sum()
        print(f"loss_fg_faces_suppress: {loss_fg_faces_suppress.detach().item():.2f}")

        if aligned_bg_faces_emb is not None:
            # Suppress background faces by pushing their embeddings towards zero.
            loss_bg_faces_suppress = (aligned_bg_faces_emb**2).mean()
            print(f"loss_bg_faces_suppress: {loss_bg_faces_suppress.detach().item():.2f}")
        else:
            loss_bg_faces_suppress = zero_losses[0]

        return loss_arcface_align, loss_fg_faces_suppress, loss_bg_faces_suppress, aligned_fg_face_bboxes, \
               aligned_fg_face_confidences, aligned_face_detected_inst_mask
