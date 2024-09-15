## Downsamples our training clips by taking less frames from the beginning of the video (where we need less info),
## and keeping more of the frames at the end, where we need to see all blade actions. 

import cv2
import subprocess as sp
import os

FFMPEG_BIN = "ffmpeg"
fps = "13"

downsample_until_frame_number = 16
downsample_by_divisor = 2

for filename in os.listdir(os.path.join(os.getcwd(), "training_quarantine")):
    if filename.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join("training_quarantine", filename))

        output_file = os.path.join('training_data', filename)
        command = [
            FFMPEG_BIN,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '640x360',
            '-pix_fmt', 'bgr24',
            '-r', fps,
            '-i', '-',
            '-an',
            '-vcodec', 'mpeg4',
            '-b:v', '5000k',
            output_file
        ]

        proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        print("Processing:", filename)

        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1
            if counter <= downsample_until_frame_number and counter % downsample_by_divisor == 0:
                proc.stdin.write(frame.tobytes())
            elif counter > downsample_until_frame_number:
                proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.stderr.close()
        print(filename + " - successful")

        # Release everything if job is finished
        cap.release()