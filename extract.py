import mediapipe as mp
from function import *
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import glob
from tqdm import tqdm

class MPObject():
    def __init__(self):
        # initialize
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        self.holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)

        WHITE_COLOR = (224, 224, 224)
        BLACK_COLOR = (0, 0, 0)
        RED_COLOR = (0, 0, 255)
        GREEN_COLOR = (0, 128, 0)
        BLUE_COLOR = (255, 0, 0)

        self.drawing_face_spec = self.mp_drawing.DrawingSpec(color=WHITE_COLOR, thickness=1, circle_radius=1)
        self.drawing_pose_spec = self.mp_drawing.DrawingSpec(color=WHITE_COLOR, thickness=3, circle_radius=3)
        self.drawing_hand_spec = self.mp_drawing.DrawingSpec(color=WHITE_COLOR, thickness=3, circle_radius=3)
        self.drawing_dot_spec = self.mp_drawing.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=3)

        self.nframes=0
        self.fps = 0
        self.interval = 0

        reset_folder('frames')

    def read_video(video_file, image_dir, image_file):
        self.fps, self.nframes, self.interval = video_2_images(video_file, image_dir, image_file)
        img = cv2.imread('frames/000000.jpg')
        cv2_imshow(img)

    def apply_mp():
        reset_folder('images')
        files = []
        for name in sorted(glob.glog('./frames/*.jpg')):
            files.append(name)
        images = [name: cv2.imread(name) for name in files]

        for name, image in tqdm(images.items()):
            results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                image = annotated_image,
                landmark_list = results.left_hand_landmarks,
                connections = self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec = self.drawing_dot_spec,
                connection_drawing_spec = self.drawing_hand_spec
            )
            self.mp_drawing.draw_landmarks(
                image = annotated_image,
                landmark_list = results.right_hand_landmarks,
                connections = self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec = self.drawing_dot_spec,
                connection_drawing_spec = self.drawing_hand_spec
            )
            self.mp_drawing.draw_landmarks(
                image = annotated_image,
                landmark_list = results.face_landmarks,
                connections = self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec = self.drawing_face_spec,
                connection_drawing_spec = self.drawing_face_spec
            )
            self.mp_drawing.draw_landmarks(
                image = annotated_image,
                landmark_list = results.pose_landmarks,
                connections = self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec = self.drawing_pose_spec,
                connection_drawing_spec = self.drawing_pose_spec
            )

            save_name = 'images/'+os.path.basename(name)
            cv2.imgwrite(save_name, annotated_image)

    def image2video(out_path):
        fps_r = self.fps/self.interval
        ! ffmpeg -y -r $fps_r -i images/%6d.jpg -vcodec libx264 -pix_fmt yuv420p -loglevel error $out_path
