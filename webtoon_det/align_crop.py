import argparse
import cv2
from anime_face_detector import create_detector
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import rotate
import torch


data_path = "/opt/ml/webtoon/webtoon_images"
face_score_threshold = 0.9  #@param {type: 'slider', min: 0, max: 1, step:0.1}
landmark_score_threshold = 0.3  #@param {type: 'slider', min: 0, max: 1, step:0.1}
show_box_score = True  #@param {'type': 'boolean'}
draw_contour = False  #@param {'type': 'boolean'}
skip_contour_with_low_score = True  #@param {'type': 'boolean'}

def get_center_eye(landmark):
    # get center of each eye
    left_eye_center = list(map(int, (landmark[12]+landmark[15])/2))[:-1]
    right_eye_center = list(map(int, (landmark[18]+landmark[21])/2))[:-1]

    return left_eye_center, right_eye_center

def get_euclidean_dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def get_angle(landmark, eps=1e-6):
    left_eye_center, right_eye_center = get_center_eye(landmark)

    left_eye_x, left_eye_y = left_eye_center[0], left_eye_center[1]
    right_eye_x, right_eye_y = right_eye_center[0], right_eye_center[1]

    if left_eye_y < right_eye_y: # counter clockwise (the bigger y-value, the lower in the image coords)
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1
    else: # clockwise
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1

    ## get a triangle
    a = get_euclidean_dist(left_eye_center, point_3rd)
    b = get_euclidean_dist(right_eye_center, left_eye_center)
    c = get_euclidean_dist(right_eye_center, point_3rd)

    # cosine rule (calculate an angle)
    cos_a = (b**2 + c**2 - a**2) / ((2*b*c)+eps)
    radian_angle = np.arccos(cos_a)
    angle = (radian_angle*180) / np.pi

    if direction == -1: # clockwise angle
        angle = 90 - angle

    return direction*angle

def rotate_bbox(bbox, d_angle, cx ,cy, h, w):
    """
    get bbox and return rotated bboxes

    - Args
        bbox: a bounding box (x1, y1, x2, y2)
        d_angle: angle by which the image is to be rotated in degrees.
        cx: x coordinate of the center of the image
        cy: y coordinate of the center of the image
        h: height of the image
        w: width of the image
    """
    # get corners
    upper_left = np.array([bbox[0], bbox[1]])
    lower_left = np.array([bbox[0], bbox[3]])
    upper_right = np.array([bbox[2], bbox[1]])
    lower_right = np.array([bbox[2], bbox[3]])
    
    corners = np.array([upper_left, upper_right, lower_right, lower_left]).reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    # get rotated bbox
    M = cv2.getRotationMatrix2D((cx,cy), d_angle, 1.0)
    
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))

    M[0,2] += (nW/2)-cx
    M[1,2] += (nH/2)-cy

    rotated = M.dot(corners.T).T.reshape(-1,8).squeeze()

    return np.round(rotated).astype(int) # tilted bbox

def get_enclosing_bbox(rotated_bbox):
    """
    get a bounding box enclosing a tilted bounding box

    - Args
        rotated_bbox: x1, y1, ..., x4, y4 (total 8 points given)
    """
    x1 = min(rotated_bbox[::2])
    y1 = min(rotated_bbox[1::2])
    x2 = max(rotated_bbox[::2])
    y2 = max(rotated_bbox[1::2])

    return [x1, y1, x2, y2]

def crop_bbox(img, bbox, x_margin=60, y_margin=100):
    """
    generate a bbox crop from a bbox information

    - Args
        img: an original image (as a numpy array)
        bbox: a bounding box info
        x_margin: a margin for x coords regarding generating a bbox
        y_margin: a margin for y coords regarding generating a bbox
    """
    h,w = img.shape[0], img.shape[1]
    x1, y1, x2, y2 = map(int, bbox)
    x_margin = round((x2-x1)*.05)
    y_margin = round((y2-y1)*.05)
    
    # margin 추가
    x1 = max(0, x1-x_margin) 
    x2 = min(x2+x_margin, w)
    y1 = max(0, y1-y_margin) 
    y2 = min(y2+y_margin, h)

    # crop
    cropped = img[y1:y2, x1:x2] # (h,w,c)
    
    return cropped

def align_face(img, landmark, bbox):
    """
    align a face based on landmark
    (applied to only a single image)
    """
    h, w, _ = img.shape
    cx, cy = w//2, h//2

    angle = get_angle(landmark)

    if abs(angle) == 0:
        return img
    
    # rotate image
    aligned_img = rotate(img, angle, reshape=True)
    # rotate box
    aligned_bbox = get_enclosing_bbox(rotate_bbox(bbox, angle, cx, cy, h, w))
    # get aligned face crop
    aligned_crop = crop_bbox(aligned_img, aligned_bbox)
    
    return aligned_crop

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = create_detector(args.model, device=device)

    img = cv2.imread(args.image_name)
    pred = detector(img)
    
    if not pred:
        print("Face Detection Failed. Try Another Image.")
        return

    bbox, landmark = pred[0]['bbox'], pred[0]['keypoints'] # choose the one with highest confidence.
    #print(bbox, landmark[:,2])
    #print(type(landmark), landmark.shape)
    if (bbox[-1] < args.face_thres) or (landmark[:,2] < args.landmark_thres).any():
        print("No Valid Face Detected. Try Another Image.")
        return

    aligned_crop = align_face(img, landmark, bbox)

    return aligned_crop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, default="/opt/ml/webtoon/webtoon_images/노답소녀/2.jpg")
    parser.add_argument("--model", type=str, default="yolov3", help="Choose the model for a detection. The available options are yolov3 or faster-rcnn.")
    parser.add_argument("--face_thres", type=float, default=0.9, help="face score threshold. The image of which confidence score smaller than this threshold would be ignored.")
    parser.add_argument("--landmark_thres", type=float, default=0.3, help="face landmark score threshold. The image of which confidence score smaller than this threshold would be ignored.")
    args = parser.parse_args()

    res = main(args)
    print("Done!")