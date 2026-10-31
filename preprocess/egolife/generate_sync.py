import json
import os
from typing import Dict, List

import pandas as pd
import pysrt
from tqdm import tqdm

BASE_DIR = "data/EgoLife"
DENSE_CAPTION_DIR = os.path.join(BASE_DIR, "EgoLifeCap/DenseCaption")
TRANSLATED_DIR = os.path.join(DENSE_CAPTION_DIR, "translated")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "EgoLifeCap/Transcript")
SYNC_DIR = os.path.join(BASE_DIR, "EgoLifeCap/Sync")

def get_captions(caption_path: str) -> List[Dict[str, str]]:
    captions = []
    with open(caption_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data["custom_id"]
            start = int(custom_id.split("-")[-2])
            end = int(custom_id.split("-")[-1])
            text = data["translated_text"]
            captions.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "type": "caption",
                }
            )
    return captions


def get_transcripts(transcript_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(transcript_path):
        return []

    subs = list(pysrt.open(transcript_path))
    hour = int(os.path.basename(transcript_path).split("_")[-1][:2])
    transcripts: List[Dict[str, str]] = []

    for sub in subs:
        lines = sub.text.split("\n")
        # English line is expected to be the last line (timestamp is separate, Chinese precedes it)
        text = lines[-1].strip() if lines else ""
        elem = {
            "start": int(f"{(hour + sub.start.hours):02d}{sub.start.minutes:02d}{sub.start.seconds:02d}"),
            "end": int(f"{(hour + sub.end.hours):02d}{sub.end.minutes:02d}{sub.end.seconds:02d}"),
            "text": text,
            "type": "transcript",
        }
        transcripts.append(elem)

    return transcripts


def find_video(files: List[str], start_time: int, end_time: int):
    for file in files:
        video_time = int(file.split("_")[-1][:-4])
        if start_time <= video_time < end_time:
            return file
    return None


def handle_time(time: int):
    if time % 10000 == 6000:
        time = time + 4000
    if time % 1000000 == 600000:
        time = time + 400000
    return time


def match_with_video(info: tuple, all_df: pd.DataFrame):
    name, date, time = info
    time = int(time)
    root_dir = os.path.join(BASE_DIR, name, date)
    
    if not os.path.isdir(root_dir):
        return []
    all_video_files = sorted([f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))])

    start_time = time
    end_time = start_time + 3000

    results = []
    while start_time < time + 1000000:
        video_file = find_video(all_video_files, start_time, end_time)
        if video_file is not None:
            cur_df = all_df[(all_df["start"] >= (start_time // 100)) & (all_df["end"] <= (end_time // 100))]
            if len(cur_df) > 0:
                data = []
                for _, row in cur_df.iterrows():
                    data.append(
                        {
                            "start": row["start"],
                            "end": row["end"],
                            "text": row["text"],
                            "type": row["type"],
                        }
                    )

                results.append({"video_file": video_file, "data": data})

        start_time = end_time
        end_time = handle_time(end_time + 3000)

    return results


def main():
    translated_files = sorted([f for f in os.listdir(TRANSLATED_DIR) if f.endswith(".jsonl")])
    for caption_file in tqdm(translated_files, desc="Syncing captions"):
        file_name = caption_file[:-6]
        idx, name, date, time = file_name.split("_")
        name = idx + "_" + name

        caption_path = os.path.join(TRANSLATED_DIR, caption_file)
        transcript_path = os.path.join(TRANSCRIPT_DIR, name, date, f"{file_name}.srt")

        captions = get_captions(caption_path)
        transcripts = get_transcripts(transcript_path)

        all_data = captions + transcripts
        all_df = pd.DataFrame(all_data)
        all_df.sort_values(by="start", inplace=True)

        results = match_with_video((name, date, time), all_df)
        os.makedirs(SYNC_DIR, exist_ok=True)
        with open(os.path.join(SYNC_DIR, f"{file_name}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
