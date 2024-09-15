from pytubefix import YouTube # This is a patched version of pytube that fixes some issues with downloading videos.
import os
import signal
import time
import traceback
import ssl
from urllib.error import HTTPError

## Timeout for use with try/except so that pytube does not freeze.
class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Create all the directories needed if they don't yet exist
directories = ['precut', 'videos', 'training_quarantine', 'more_training_data', 'final_training_clips', 'optical_flow',
               'preinception_data', 'final_training_data', 'training_data']
for dirs in directories:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

text_file = open("foil_videos.txt", "r")
vids = text_file.read().split('\n')
text_file.close()

# Loop through all the videos, download them, and put them in the precut folder.
for i in range(len(vids)):
    try:
        url = vids[i].split('&')[0]  # Remove any query parameters

        with Timeout(600):
            start = time.time()
            yt = YouTube(url)

            # Use progressive streams to avoid separate audio/video files
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

            if stream:
                stream.download(output_path=os.getcwd() + '/precut/')
                print("Downloaded:", vids[i], "   ", (time.time() - start), "s")
            else:
                print("No suitable streams found for:", vids[i])

    except HTTPError as e:
        print(f"HTTPError for {vids[i]}: {e}")
    except Timeout.Timeout:
        print(f"Timeout - {vids[i]}")
    except Exception as e:
        traceback.print_exc()
        print("Failed -", vids[i])