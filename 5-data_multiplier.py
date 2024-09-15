## Performs data-augmentation by flipping all clips horizontally. 

import cv2
import subprocess as sp
import os

FFMPEG_BIN = "ffmpeg"
fps = "13"

for filename in os.listdir(os.path.join(os.getcwd(), "training_data")):
    if filename.endswith(".mp4"):
        cap = cv2.VideoCapture(os.path.join("training_data", filename))

        # Generate new filename for flipped video
        if filename.startswith('L'):
            new_filename = 'R' + filename.lstrip('L')
        elif filename.startswith('R'):
            new_filename = 'L' + filename.lstrip('R')
        else:
            new_filename = filename

        output_file = os.path.join('more_training_data', new_filename.replace('.mp4', '-flipped.mp4'))

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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        print("stderr")
        proc.stderr.close()
        print("Processing successful")

        # Release everything if job is finished
        cap.release()