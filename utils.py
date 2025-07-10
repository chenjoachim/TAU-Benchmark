import pydub
import gdown
import yt_dlp
import os
import shutil
import uuid

SAMPLING_RATE = 16000  # Hz


def download_from_google_drive(
    url, format="mp3", output_path="data/audio", output_id=None
) -> str:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not output_id:
        output_id = str(uuid.uuid4().hex)[
            :8
        ].upper()  # Generate a unique ID if not provided
    # Extract the file ID from the URL
    file_id = url.split("/")[-2]
    output_file = os.path.join(output_path, f"{output_id}.{format}")

    os.makedirs("tmp", exist_ok=True)

    try:
        downloaded_file = gdown.download(
            f"https://drive.google.com/uc?id={file_id}", f"tmp/", quiet=True
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return ""

    audio = pydub.AudioSegment.from_file(downloaded_file)
    audio = audio.set_frame_rate(SAMPLING_RATE)
    audio.export(output_file, format=format)
    os.remove(downloaded_file)

    return output_file


def download_from_yt(
    url, format="mp3", output_path="data/audio", output_id=None
) -> str:
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg or provide the path using ffmpeg_path parameter."
        )

    if not output_id:
        output_id = str(uuid.uuid4().hex)[:8].upper()

    output_file = os.path.join(output_path, output_id)

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
        return f"{output_file}.{format}"
    except Exception as e:
        print(f"Download failed: {e}")
        return ""


def crop_audio(
    audio_path: str, start_ms: int, end_ms: int, output_format="mp3", max_length=15000
):
    """Crop audio file from start_ms to end_ms."""
    audio = pydub.AudioSegment.from_file(audio_path)
    if start_ms < 0:
        start_ms = 0
    if end_ms < 0 or end_ms - start_ms > max_length:
        end_ms = start_ms + max_length
    cropped_audio = audio[start_ms : start_ms + max_length]
    cropped_audio.export(audio_path, format=output_format)
    return audio_path, start_ms, end_ms


if __name__ == "__main__":
    # Example usage
    download_from_yt("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    download_from_google_drive(
        "https://drive.google.com/file/d/1gPi5zsAtaqaEbzSpBuT-Wrfii5EBBvfi/view?usp=sharing"
    )
