import scipy.io as sio
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from time import time

import pickle


def get_project_root():
    '''
    :return:  path without slash in the end.
    '''
    path = f'{Path(__file__).parent.parent}/'
    return path


def get_issia_dataset_path():
    return f'{get_project_root()}datasets/issia/'


def get_soccer_player_detection_dataset_path():
    return f'{get_project_root()}datasets/SoccerPlayerDetection_bmvc17_v1/'


def get_world_cup_2014_scc_dataset_path():
    return f'{get_project_root()}datasets/world_cup_2014_scc/'


def get_camera_estimator_files_path():
    return f'{get_project_root()}modules/Camera_Estimator/files/'


def get_siamese_model_path():
    return f'{get_project_root()}models/generated_models/siamese/'


def get_two_pix2pix_model_path():
    return f'{get_project_root()}models/generated_models/two_pix2pix/'


def get_footandball_model_path():
    return f'{get_project_root()}models/generated_models/foot_and_ball/'


def get_yolov3_model_path():
    return f'{get_project_root()}models/generated_models/yolo_v3/'


def get_binary_court():
    return sio.loadmat(f'{get_camera_estimator_files_path()}worldcup2014.mat')


def save_model(model_components, history, filename):
    """Save trained model along with its optimizer and training, plottable history."""
    state = {}
    for key in model_components.keys():
        component = model_components[key]
        state[key] = component.state_dict()
    state['history'] = history
    torch.save(state, filename)


def load_pickle_file(filename):
    data = None
    with open(filename, 'rb') as file:
        while True:
            try:
                data = pickle.load(file)
            except EOFError:
                break
        file.close()
        return data


def load_text_file(filename):
    text = None
    try:
        with open(filename, 'r') as file:
            text = file.read()
    except Exception:
        print(f"File not accessible: {filename}")
    return text


def save_to_pickle_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()


def show_image(img_list, msg_list=None):
    """
    Display N images. Esc char to close window. For debugging purposes.
    :param img_list: A list with images to be displayed.
    :param msg_list: A list with title for each image to be displayed. If not None, it has to be of equal length to
    the image list.
    :return:
    """
    if not isinstance(img_list, list):
        return 'Input is not a list.'

    if msg_list is None:
        msg_list = [f'{i}' for i in range(len(img_list))]
    else:
        msg_list = [f'{msg}' for msg in msg_list]

    for i in range(len(img_list)):
        cv2.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
        cv2.destroyWindow(msg)


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def draw_image_with_bboxes(img, bboxes):
    out = np.array(img)
    for b in bboxes:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
    return out


def draw_image_with_bboxes2(img, bboxes):
    out = np.array(img)
    for b in bboxes:
        x1, y1, w, h = b
        cv2.rectangle(out, (int(x1), int(y1)), (int(x1 + h), int(y1 + w)), (255, 0, 0), 2)
    return out


def concatenate_multiple_images(*images):
    img_height = images[0].shape[0]
    img_delimiter = np.ones((img_height, 2, 3), dtype=np.uint8) * 128
    image_to_be_saved = img_delimiter
    for im in images:
        image_to_be_saved = np.concatenate((image_to_be_saved, im, img_delimiter), axis=1)
    return image_to_be_saved


def normalize(x, axis=0):
    return (x - np.min(x, axis=axis)) / (np.max(x, axis=axis) - np.min(x, axis=axis))


def convert_list_to_int(_list):
    return [int(v) for v in _list]


def export_image_boxes_from_image(frame, boxes):
    boxes = [convert_list_to_int(l) for l in boxes]
    boxes = [export_image_from_box(frame, box) for box in boxes]
    return boxes


def export_image_from_box(frame, box):
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2, :]


def logging_time(func):
    """Decorator that logs time"""

    def logger(*args, **kwargs):
        """Function that logs time"""
        start = time()
        output = func(*args, **kwargs)
        print(f'Exec. time - {func.__name__}: {time() - start:.5f}')
        return output
    return logger
