import cv2
from imutils.video import FPS
from util import utils


def get_frames_from_video(video_file, num_of_frames):
    video_handler = VideoHandler(file=video_file, output_resolution=(1280, 720))
    image_data = [video_handler.get_frame() for i in range(num_of_frames)]
    video_handler.release()
    return image_data


class VideoHandler:

    def __init__(self, file, output_resolution=None):
        self.video_stream = cv2.VideoCapture(file)
        self.video_resolution = (
                int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        self.frame_count = 0
        self.fps = FPS().start()

        if output_resolution is None:
            self.output_resolution = self.video_resolution
        else:
            self.output_resolution = output_resolution

        _, self.current_frame = self.video_stream.read()
        self.next_frame = None
        if self.current_frame is not None:
            _, self.next_frame = self.video_stream.read()
        else:
            print('No video file.')

        self.codec = None
        self.output_video_stream = None
        self.play_video = True

    def start_recording(self, output_file, recording_resolution):
        self.codec = cv2.VideoWriter_fourcc(*'XVID')

        output_fps = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.output_video_stream = cv2.VideoWriter(output_file, self.codec, output_fps, recording_resolution)

    def has_frames(self):
        return self.play_video and self.next_frame is not None

    def get_frame(self):
        self.current_frame = self.next_frame
        _, self.next_frame = self.video_stream.read()
        self.fps.update()
        self.frame_count += 1
        return cv2.resize(self.current_frame, self.output_resolution)

    def _record(self, frame):
        self.output_video_stream.write(frame)

    def release(self):
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        self.video_stream.release()
        if self.output_video_stream is not None:
            self.output_video_stream.release()

    def show_image(self, window_title, frame):
        if self.output_video_stream is not None:
            self._record(frame)
        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.play_video = False
        if key == ord('p'):
            cv2.waitKey(-1)
