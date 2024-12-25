import cv2
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms








def process_latent_id(image, netArc, device, crop_size=224):
    """
    Process the input image (crop, align, normalize, tensor conversion) and extract the latent ID from the ArcFace model.
    
    :param image: The input image (BGR format).
    :param netArc: The ArcFace model to extract latent features from.
    :param device: The device (CPU or CUDA) to run the model on.
    :param crop_size: The size to crop and align the face (default is 224x224).
    :return: The latent ID tensor from the ArcFace model.
    """
    source_crop_align_image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    to_tensor_transform = transforms.ToTensor()
    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Apply the transformations
    source_crop_align_image_pil = to_tensor_transform(source_crop_align_image_pil)
    source_crop_align_image_pil = normalize_transform(source_crop_align_image_pil)

    # Add batch dimension [1, C, H, W]
    source_crop_align_image_pil = source_crop_align_image_pil.unsqueeze(0)
    source_crop_align_image_pil = source_crop_align_image_pil.to(device)
    
    # Downsample to 112x112 for ArcFace
    source_downsample = F.interpolate(source_crop_align_image_pil, size=(112, 112))
    
    # Pass through ArcFace model to get latent ID
    latent_id = netArc(source_downsample)
    latent_id = F.normalize(latent_id, p=2, dim=1)
    
    return latent_id




def process_image_tensor(image):
    """
    Process an image (BGR to RGB, to Tensor, and normalize) and move to GPU.
    
    :param image: The input image in BGR format.
    :return: A Tensor suitable for GPU processing.
    """
    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(image_rgb)
    img_tensor = tensor.transpose(0, 1).transpose(0, 2).contiguous()  # Reorder the dimensions to [C, H, W]
    img_tensor = img_tensor.float().div(255)
    img_tensor = img_tensor[None, ...].cuda()

    return img_tensor