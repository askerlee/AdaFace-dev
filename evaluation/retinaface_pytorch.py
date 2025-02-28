from typing import List
import numpy as np
from retinaface.pre_trained_models import get_model
from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F

# Copied from deepface/models/Detector.py
class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        nose: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
    ):
        """
        Initialize a Face object.

        Args:
            x (int): The x-coordinate of the top-left corner of the bounding box.
            y (int): The y-coordinate of the top-left corner of the bounding box.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.
            left_eye (tuple): The coordinates (x, y) of the left eye with respect to
                the person instead of observer. Default is None.
            right_eye (tuple): The coordinates (x, y) of the right eye with respect to
                the person instead of observer. Default is None.
            confidence (float, optional): Confidence score associated with the face detection.
                Default is None.
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose = nose
        self.mouth_left  = None
        self.mouth_right = None
        self.confidence  = confidence

class RetinaFaceClient(nn.Module):
    def __init__(self, device='cuda'):
        super(RetinaFaceClient, self).__init__()
        # We have called torch.cuda.set_device(opt.gpu) in stable_txt2img.py, so to("cuda") 
        # will put the model on the correct GPU.
        self.model = get_model("biubug6", max_size=1024, device=device)

    def detect_faces(self, img: np.ndarray, T=20) -> List[FacialAreaRegion]:
        """
        Detect and align face with retinaface

        Args:
            img (np.ndarray): pre-loaded image as numpy array,
            T: minimum size of the face (the height or the width) to be detected

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        # predict_jsons is wrapped with torch.no_grad().
        objs = self.model.predict_jsons(img, confidence_threshold=0.9)
        H, W = img.shape[:2]

        '''
            {'bbox': [78.36, 113.11, 276.53, 375.64], 'score': 1.0, 
            'landmarks': [[130.65, 218.37], [224.65, 214.67], [182.26, 260.41], [140.13, 305.17], [227.34, 301.35]]}        
        '''
        for identity in objs:
            detection = identity["bbox"]
            if len(detection) != 4:
                # No face detected
                continue

            # clip detection box to image size
            y1, y2 = max(detection[1], 0), min(detection[3], H)
            x1, x2 = max(detection[0], 0), min(detection[2], W)
            h = y2 - y1
            w = x2 - x1

            if h <= T or w <= T:
                # Face too small
                continue

            # retinaface sets left and right eyes with respect to the person
            # The landmark seems to be mirrored compared with deepface detectors.
            # Returns 5-point facial landmarks: right eye, left eye, nose, right mouth, left mouth
            left_eye  = identity["landmarks"][1]
            right_eye = identity["landmarks"][0]
            nose      = identity["landmarks"][2]

            # eyes are list of float, need to cast them tuple of int
            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)
            #print("left_eye: ", left_eye)
            #print("right_eye: ", right_eye)

            confidence = identity["score"]

            facial_area = FacialAreaRegion(
                x=x1,
                y=y1,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                confidence=confidence,
            )

            resp.append(facial_area)

        return resp

    # Find facial areas of given image tensors and crop them.
    # images_ts: typically [BS, 3, 512, 512] from diffusion (could be any sizes).
    # image_ts: [3, 512, 512].
    # Output: [BS, 3, 128, 128] (cropped faces resized to 128x128), face_detected_inst_mask, face_bboxes
    def crop_faces(self, images_ts, out_size=(128, 128), T=20, bleed=0):
        face_detected_inst_mask = []
        fg_face_bboxes      = []
        fg_face_crops       = []
        bg_face_crops_flat  = []
    
        for i, image_ts in enumerate(images_ts):
            # [3, H, W] -> [H, W, 3]
            image_np = image_ts.detach().cpu().numpy().transpose(1, 2, 0)
            # [-1, 1] -> [0, 255]
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)

            # .detect_faces() doesn't require grad. So we convert the image tensor to numpy.
            faces = self.detect_faces(image_np, T=T)
            if len(faces) == 0:
                face_crop = image_ts
                face_crop = F.interpolate(face_crop.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False)
                fg_face_crops.append(face_crop)
                fg_face_bboxes.append((0, 0, image_ts.shape[2], image_ts.shape[1]))
                # No face detected
                face_detected_inst_mask.append(0)
                continue
            else:
                face_bboxes = []
                # Find the largest facial area
                for face in faces:
                    x = face.x
                    y = face.y
                    w = face.w
                    h = face.h

                    y_start = max(0, int(y + bleed))
                    y_end   = min(image_ts.shape[1], int(y + h - bleed))
                    x_start = max(0, int(x + bleed))
                    x_end   = min(image_ts.shape[2], int(x + w - bleed))

                    if y_start + T >= y_end or x_start + T >= x_end:
                        continue
                    
                    facial_area = (y_end - y_start) * (x_end - x_start)
                    face_bboxes.append((facial_area, x_start, y_start, x_end, y_end))

                if len(face_bboxes) == 0:
                    # After trimming bleed pixels, no faces are >= T. So this instance is failed.
                    face_crop = image_ts
                    face_crop = F.interpolate(face_crop.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False)
                    fg_face_crops.append(face_crop)
                    fg_face_bboxes.append((0, 0, image_ts.shape[2], image_ts.shape[1]))
                    # No face detected
                    face_detected_inst_mask.append(0)
                    continue

                # Sort face_bboxes by the facial area in descending order
                face_bboxes = sorted(face_bboxes, key=lambda x: x[0], reverse=True)
                # Remove the facial area from the tuple
                face_bboxes = [x[1:] for x in face_bboxes]

                face_crops = []
                for x_start, y_start, x_end, y_end in face_bboxes:
                    # Extract detected face
                    # Crop on the input tensor, so that computation graph is preserved.
                    face_crop = image_ts[:, y_start:y_end, x_start:x_end]
                    # resize to (1, 3, 128, 128)
                    face_crop = F.interpolate(face_crop.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False)
                    face_crops.append(face_crop)

                # Only keep the coords of the largest face of each instance in fg_face_bboxes.
                # We don't care about the coords of the bg faces, so don't store them.
                fg_face_bboxes.append(face_bboxes[0])
                # Keep all face crops in fg_face_crops.
                fg_face_crops.append(face_crops[0])
                face_detected_inst_mask.append(1)
                bg_face_crops_flat.extend(face_crops[1:])

        # fg_face_bboxes: long tensor of [BS, 4].
        fg_face_bboxes      = torch.tensor(fg_face_bboxes, device=images_ts.device)
        # face_detected_inst_mask: binary tensor of [BS]
        face_detected_inst_mask = torch.tensor(face_detected_inst_mask, device=images_ts.device)

        # fg_face_crops: [BS, 3, 128, 128], BS1 <= BS
        fg_face_crops = torch.cat(fg_face_crops, dim=0)

        if len(bg_face_crops_flat) == 0:
            bg_face_crops_flat = None
        else:
            # bg_face_crops_flat: [N, 3, 128, 128], N is the total number of bg face crops.
            bg_face_crops_flat  = torch.cat(bg_face_crops_flat, dim=0)
        
        return fg_face_crops, bg_face_crops_flat, fg_face_bboxes, face_detected_inst_mask

