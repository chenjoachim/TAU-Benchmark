import random
import time
import pydub
import gdown
import yt_dlp
import os
import shutil
import uuid

SAMPLING_RATE = 44100  # Hz


def download_from_google_drive(
    url, format="mp3", output_path="data/audio", output_id=None, audio_idx=None
) -> tuple[str, bool]:
    os.makedirs(output_path, exist_ok=True)

    if not output_id:
        output_id = str(uuid.uuid4().hex)[
            :8
        ].upper()  # Generate a unique ID if not provided
    # Extract the file ID from the URL
    file_id = url.split("/")[-2]
    if audio_idx:
        output_file = os.path.join(output_path, f"{output_id}_{audio_idx}.{format}")
    else:
        output_file = os.path.join(output_path, f"{output_id}.{format}")

    os.makedirs("tmp", exist_ok=True)

    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping download.")
        return output_file, False

    for attempt in range(5):
        if attempt > 0:
            print(f"Retrying download from Google Drive (attempt {attempt + 1})")
        
        
        try:
            downloaded_file = gdown.download(
                f"https://drive.google.com/uc?id={file_id}", f"tmp/", quiet=True
            )
            if not downloaded_file:
                raise Exception("Download failed, file not found.")
            break  # Exit loop if download is successful
            
        except Exception as e:
            if attempt == 4:
                print(f"Failed to download after 5 attempts: {e}")
                return "", False
            time.sleep(3 ** (attempt + 1) + random.uniform(0, 1))  # Wait before retrying

    audio = pydub.AudioSegment.from_file(downloaded_file)
    audio = audio.set_frame_rate(SAMPLING_RATE)
    audio.export(output_file, format=format)
    os.remove(downloaded_file)

    return output_file, True


def download_from_yt(
    url, format="mp3", output_path="data/audio", output_id=None, audio_idx=None
) -> tuple[str, bool]:
    os.makedirs(output_path, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg or provide the path using ffmpeg_path parameter."
        )

    if not output_id:
        output_id = str(uuid.uuid4().hex)[:8].upper()

    if audio_idx:
        output_file = os.path.join(output_path, f"{output_id}_{audio_idx}")
    else:
        output_file = os.path.join(output_path, output_id)

    if os.path.exists(f"{output_file}.{format}"):
        print(f"File {output_file}.{format} already exists, skipping download.")
        return f"{output_file}.{format}", False

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": format,
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar",
            str(SAMPLING_RATE),
        ],
        "outtmpl": output_file,
        "noplaylist": True,
        "quiet": True,
        "ffmpeg_location": ffmpeg_path,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{output_file}.{format}", True
    except Exception as e:
        print(f"Download failed: {e}")
        return "", False


def download_from_curl(
    url, format="mp3", output_path="data/audio", output_id=None, audio_idx=None
) -> tuple[str, bool]:
    os.makedirs(output_path, exist_ok=True)

    if not output_id:
        output_id = str(uuid.uuid4().hex)[:8].upper()

    if audio_idx:
        output_file = os.path.join(output_path, f"{output_id}_{audio_idx}.{format}")
    else:
        output_file = os.path.join(output_path, f"{output_id}.{format}")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping download.")
        return output_file, False

    try:
        os.system(f"curl -sL {url} -o {output_file}")
        return output_file, True
    except Exception as e:
        print(f"Download failed: {e}")
        return "", False


def crop_audio(
    audio_path: str, start_ms: int, end_ms: int, output_format="mp3", max_length=15000, need_crop=True
):
    """Crop audio file from start_ms to end_ms."""
    
    audio = pydub.AudioSegment.from_file(audio_path)
    if start_ms < 0:
        start_ms = 0
    if end_ms < 0 or end_ms - start_ms > max_length:
        end_ms = max(start_ms + max_length, len(audio))
    if not need_crop:
        # Already cropped or no need to crop
        return audio_path, start_ms, end_ms
    cropped_audio = audio[start_ms : end_ms]
    cropped_audio.export(audio_path, format=output_format)
    return audio_path, start_ms, end_ms


if __name__ == "__main__":
    # Example usage
    download_from_yt("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    download_from_google_drive(
        "https://drive.google.com/file/d/1gPi5zsAtaqaEbzSpBuT-Wrfii5EBBvfi/view?usp=sharing"
    )
