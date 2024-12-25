import cv2

from util import face_align_ffhqandnewarc as face_align


def crop_and_align_face(app, image, crop_size):
    """
    Crop and align the face from the input image using InsightFace and a specified crop size.
    Also returns the affine transformation matrix used for alignment.
    
    :param app: The initialized InsightFace FaceAnalysis object.
    :param image: The input image (should be in BGR format).
    :param crop_size: The desired output crop size (both width and height).
    :return: A tuple containing the aligned face image and the affine transformation matrix.
    """
    
    det_result = app.get(image, crop_size)
    
    if len(det_result) == 0:
        raise ValueError("No face detected in the input image.")

    M, _ = face_align.estimate_norm(det_result[0]["kps"], crop_size)
    aligned_face = cv2.warpAffine(image, M, (crop_size, crop_size), borderValue=0.0)
    
    return aligned_face, M


