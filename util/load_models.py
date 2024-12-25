"""
load_models.py

This script contains functions to load and initialize various models. It includes the following models:
- ArcFace: A facial recognition model used for face verification and identification.
- FaceSwap: A model for performing face swapping, requiring a pre-trained ArcFace model.
- InsightFace: A face detection model from the InsightFace library, used for detecting faces in images or videos.

Functions:
1. load_netArc(Arc_path, device): Loads the ArcFace model from a checkpoint and transfers it to the specified device (CPU or GPU).
2. load_face_swap_model(device, netArc, opt): Initializes the face swap model with the given ArcFace model and configuration.
3. load_insightface_model(device): Initializes the InsightFace model (FaceAnalysis) and prepares it for face detection.

The device parameter is used to specify whether the model should run on CPU or GPU. For GPU, the device should be in the format 'cuda:<index>' (e.g., 'cuda:0' for the first GPU).

By using these functions, users can load the necessary models into memory, ready for inference or other tasks such as face detection, face swapping, and recognition.

Author: JethroChow
"""

import torch

import insightface
from insightface.app import FaceAnalysis

from models.fs_model import fsModel



def load_netArc(Arc_path, device):
    """
    Load the ArcFace model and transfer it to the specified device.
    
    :param Arc_path: Path to the ArcFace model checkpoint file.
    :param device: PyTorch device (e.g., 'cuda:0' or 'cpu').
    :return: Loaded ArcFace model.
    """
    netArc = torch.load(Arc_path, map_location=torch.device("cpu"))
    netArc = netArc.to(device)
    netArc.eval()
    
    return netArc



def load_face_swap_model(device, netArc, opt):
    """
    Load and initialize the face swap model (fsModel).
    
    :param device: PyTorch device (e.g., 'cuda:0' or 'cpu').
    :param netArc: Pre-trained ArcFace model.
    :param opt: Configuration object containing additional parameters required by the model.
    :return: Initialized face_swap_model instance.
    """
    from models.fs_model import fsModel

    face_swap_model = fsModel()
    face_swap_model.initialize(device, netArc, opt)
    
    return face_swap_model



def load_insightface_model(device, model_name="antelopev2"):
    """
    Load and initialize the InsightFace model (FaceAnalysis) with the specified device.

    :param device: PyTorch device (e.g., 'cuda:0', 'cuda:1', or 'cpu').
    :param model_name: The name of the model (default is 'antelopev2').
    :return: Prepared FaceAnalysis instance.
    """
    device_str = str(device)

    if device_str == 'cpu':
        ctx_id = -1
    elif device_str.startswith('cuda'):
        # Extract device index from 'cuda:<index>' (e.g., 'cuda:0', 'cuda:1')
        ctx_id = int(device_str.split(':')[1]) if ':' in device_str else 0
    else:
        raise ValueError(f"Unsupported device type: {device_str}")
        
    app = FaceAnalysis(name=model_name, root='./checkpoints')
    app.prepare(ctx_id=ctx_id, det_thresh=0.6, det_size=(640,640))
    
    return app