from typing import List
import numpy as np
from deepface.models.Detector import Detector, FacialAreaRegion
from retinaface.pre_trained_models import get_model
from PIL import Image

# pylint: disable=too-few-public-methods
class RetinaFaceClient(Detector):
    def __init__(self):
        # We have called torch.cuda.set_device(opt.gpu) in stable_txt2img.py, so to("cuda") 
        # will put the model on the correct GPU.
        self.model = get_model("resnet50_2020-07-20", max_size=1024, device='cuda')

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
