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
        self.confidence = confidence

class RetinaFaceClient(nn.Module):
    def __init__(self, device='cuda'):
        super(RetinaFaceClient, self).__init__()
        # We have called torch.cuda.set_device(opt.gpu) in stable_txt2img.py, so to("cuda") 
        # will put the model on the correct GPU.
        self.model = get_model("biubug6", max_size=1024, device=device)

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with retinaface

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        objs = self.model.predict_jsons(img, confidence_threshold=0.9)

        for identity in objs:
            detection = identity["bbox"]
            if len(detection) != 4:
                # No face detected
                continue

            y = detection[1]
            h = detection[3] - y
            x = detection[0]
            w = detection[2] - x

            # retinaface sets left and right eyes with respect to the person
            # The landmark seems to be mirrored compared with deepface detectors.
            # Returns 5-point facial landmarks: right eye, left eye, nose, right mouth, left mouth
            left_eye = identity["landmarks"][1]
            right_eye = identity["landmarks"][0]

            # eyes are list of float, need to cast them tuple of int
            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)
            #print("left_eye: ", left_eye)
            #print("right_eye: ", right_eye)

            confidence = identity["score"]

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )

            resp.append(facial_area)

        return resp

    # Find facial areas of given image tensors and crop them.
    # Output: [BS, 3, 128, 128]
    def crop_faces(self, images_ts, out_size=(128, 128)):
        face_crops = []
        failed_indices = []

        for i, image_ts in enumerate(images_ts):
            # [3, H, W] -> [H, W, 3]
            image_np = image_ts.cpu().numpy().transpose(1, 2, 0)
            # [-1, 1] -> [0, 255]
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)

            # .detect_faces() doesn't require grad.
            facial_areas = self.detect_faces(image_np)
            if len(facial_areas) == 0:
                # No face detected
                failed_indices.append(i)
                continue
            # Only use the first detected face.
            facial_area  = facial_areas[0]
            x = facial_area.x
            y = facial_area.y
            w = facial_area.w
            h = facial_area.h

            # Extract detected face without alignment
            # Crop on the input tensor, so that computation graph is preserved.
            face_crop = image_ts[:, int(y) : int(y + h), int(x) : int(x + w)]
            # resize to (1, 3, 128, 128)
            face_crop = F.interpolate(face_crop.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False)
            face_crops.append(face_crop)
        
        if len(face_crops) == 0:
            return None, failed_indices
        
        face_crops = torch.cat(face_crops, dim=0)
        return face_crops, failed_indices

