from pytubefix import Playlist
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

def save_playlist_links(playlist_urls, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Iterate over each playlist URL
        for playlist_url in playlist_urls:
            try:
                # Create a Playlist object for each URL
                playlist = Playlist(playlist_url)

                print(f"Fetching links from playlist: {playlist.title}")

                # Write each video's URL to the file
                for video in playlist.videos:
                    file.write(video.watch_url + '\n')

                print(f"Saved {len(playlist.video_urls)} links from {playlist.title}")

            except Exception as e:
                print(f"Failed to process playlist: {playlist_url}\nError: {e}")

    print(f"All links saved to {output_file}")


# Define YouTube playlist URLs
playlist_urls = [
    'https://www.youtube.com/playlist?list=PL_pQQho0KExyKIiybGuSbwqhJtyMWWXBZ', # Milano 2023 Men's Foil Individual Word Cup
    'https://www.youtube.com/playlist?list=PL_pQQho0KExwQU4aN2RxG5sTYK2OKB4Wb', # 2022 Bonn GER Men's Foil Individual World Cup
]

output_file = 'foil_videos.txt'

save_playlist_links(playlist_urls, output_file)