# Calculate optical flow, overlay it on the clip and save it as matrices instead of the mp4 files we've been using so far.

import os
import numpy as np
import cv2
import hickle as hkl
import subprocess as sp

# Move files from \training_data and \more_training_data to final_training_clips
for filename in os.listdir('training_data'):
    os.rename(os.path.join('training_data', filename), os.path.join('final_training_clips', filename))

for filename in os.listdir('more_training_data'):
    os.rename(os.path.join('more_training_data', filename), os.path.join('final_training_clips', filename))

# Create a folder /already_optical_flowed to move files to after they've been processed if not already present
if not os.path.exists('final_training_clips/already_optical_flowed'):
    os.makedirs('final_training_clips/already_optical_flowed')

def label_to_one_hot(label):
    if label == 'L':
        return (1, 0, 0)
    elif label == 'T':
        return (0, 1, 0)
    elif label == 'R':
        return (0, 0, 1)


def writeOpticalFlowToVideo(video_string):
    cap = cv2.VideoCapture(video_string)
    cap.set(cv2.CAP_PROP_FPS, 10000)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(cap.grab())

    ret, frame1 = cap.read()
    height, width, depth = frame1.shape
    height_end = height - height // 7  # Avoid fencers' names/countries
    print(height_end)
    height_start = 0

    frame1 = np.concatenate((frame1[height_start:height_end, :, :], frame1[height_end + height // 14 - 10:, :, :]),
                            axis=0)
    print(frame1.shape)

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    fps = "13"
    FFMPEG_BIN = "ffmpeg"

    output_file = os.path.join('optical_flow',
                               video_string.replace('final_training_clips/', "").replace('.mp4', '') + 'move' + '.mp4')
    command = [
        FFMPEG_BIN,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '640x345',
        '-pix_fmt', 'bgr24',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        output_file
    ]

    frames_till_video_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_frame = 2 if frames_till_video_end == 23 else 1

    print(frames_till_video_end)
    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    cv2.imshow("frame", frame1)
    subcount = 1

    while subcount <= frames_till_video_end - last_frame:
        print(subcount)
        ret, frame2 = cap.read()
        frame2 = np.concatenate((frame2[height_start:height_end, :, :], frame2[height_end + height // 14 - 10:, :, :]),
                                axis=0)
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        print(flow.shape)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        output = frame2.copy()
        gray_image = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        output[:, :, 0] = gray_image
        output[:, :, 1] = gray_image
        output[:, :, 2] = gray_image

        alpha = 0.45
        cv2.addWeighted(bgr, alpha, output, 1 - alpha, 0, output)

        proc.stdin.write(output.tobytes())

        output = output.reshape(-1, output.shape[0], output.shape[1], output.shape[2])
        if subcount == 1:
            to_save = output
        else:
            to_save = np.concatenate((to_save, output), axis=0)

        subcount += 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', bgr)

        prvs = next_frame

    proc.stdin.close()
    proc.stderr.close()
    cap.release()
    cv2.destroyAllWindows()
    to_save = np.expand_dims(to_save, axis=0)
    label = video_string.split('/')[1][0]
    label = np.expand_dims(label_to_one_hot(label), axis=0)

    print(label)
    print(to_save.shape, "to save shape")
    return to_save, label

#######################################################################################################################################
data_created = 0
data_saved = 0
data_saved_previously = []
for filename in os.listdir(os.getcwd() + "/preinception_data"):
    if filename.endswith(".hkl"):
        number = filename.replace(".hkl", '').split('-')[1]
        data_saved_previously.append(number)
if data_saved_previously:
    data_saved = int(max(data_saved_previously))
data_saved += 1
print("Largest Number Found", data_saved)
#######################################################################################################################################

for filename in os.listdir('final_training_clips'):
    print(filename)
    if filename.endswith(".mp4"):
        print(filename)
        output, label = writeOpticalFlowToVideo(os.path.join("final_training_clips", filename))
        os.rename(os.path.join("final_training_clips", filename),
                  os.path.join("final_training_clips/already_optical_flowed", filename))

        if data_created == 0:
            train_set = output
            train_labels = label
        else:
            train_set = np.concatenate((train_set, output), axis=0)
            train_labels = np.concatenate((train_labels, label), axis=0)

        data_created += 1
        #if data_created % 100 == 0:
        hkl.dump(train_set, 'preinception_data/train_set-' + str(data_saved) + '.hkl', mode='w', compression='gzip',
                 compression_opts=9)
        hkl.dump(train_labels, 'final_training_data/train_labels-' + str(data_saved) + '.hkl', mode='w',
                 compression='gzip', compression_opts=9)
        print('################### DATA SAVED', data_saved)
        data_saved += 1
        train_set = output  # Reset for next batch
        data_created = 0

#print(train_set.shape)
#print(train_labels.shape)
#hkl.dump(train_set, 'preinception_data/train_set-' + str(data_saved) + '.hkl', mode='w', compression='gzip',
#         compression_opts=9)
#hkl.dump(train_labels, 'final_training_data/train_labels-' + str(data_saved) + '.hkl', mode='w', compression='gzip',
#        compression_opts=9)