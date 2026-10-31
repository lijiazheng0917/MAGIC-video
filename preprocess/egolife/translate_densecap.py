import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pysrt
from tqdm import tqdm

from worldmm.llm import LLMModel

DEFAULT_TRANSLATION_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = "You are a helpful assistant that translates text from Chinese to English. Answer in translated text only."


def build_record(name: str, date: str, start: str, end: str, idx: int, translation: str) -> dict:
    return {
        "custom_id": f"{idx}-{name}-{date}-{start}-{end}",
        "translated_text": translation
    }


def translate(input_path: str, output_path: str, model_name: str) -> None:
    """
    Translate a single SRT file or every SRT file under a directory (e.g., DenseCaption/A1_JAKE/DAY*).
    - input_path: path to an SRT file or a directory containing DAY folders.
    - output_path: output file if input_path is a file, otherwise output directory for all translated JSONLs.
    """
    model = LLMModel(model_name=model_name)

    def _translate_file(in_file: str, out_file: str) -> None:
        subs = list(pysrt.open(in_file))
        name = in_file.split("/")[-3]
        date = in_file.split("/")[-2]
        hour = int(os.path.basename(in_file).split("_")[-1][:2])

        def _translate_one(idx: int, text: str, start: str, end: str):
            translation = model.generate([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ])
            return idx, build_record(name, date, start, end, idx, translation)

        futures = []
        with ThreadPoolExecutor() as executor:
            for idx, sub in enumerate(subs, start=1):
                start = f"{(hour + sub.start.hours):02d}{sub.start.minutes:02d}{sub.start.seconds:02d}"
                end = f"{(hour + sub.end.hours):02d}{sub.end.minutes:02d}{sub.end.seconds:02d}"
                futures.append(executor.submit(_translate_one, idx, sub.text, start, end))

            results = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Translating {os.path.basename(in_file)}"):
                idx, record = future.result()
                results[idx] = record

        ordered_records = [results[i] for i in sorted(results.keys())]

        parent_dir = os.path.dirname(out_file) or "."
        os.makedirs(parent_dir, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for record in ordered_records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        day_dirs = sorted([d for d in os.listdir(input_path) if d.startswith("DAY")])
        
        file_pairs = []
        for day in day_dirs:
            day_path = os.path.join(input_path, day)
            if not os.path.isdir(day_path):
                continue

            files = [f for f in os.listdir(day_path) if f.endswith(".srt")]
            for file in files:
                input_file = os.path.join(day_path, file)
                output_file = os.path.join(output_path, file.replace(".srt", ".jsonl"))
                file_pairs.append((input_file, output_file))
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_translate_file, in_f, out_f): in_f for in_f, out_f in file_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating files"):
                future.result()
    else:
        _translate_file(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate DenseCaption SRT files to JSONL.")
    parser.add_argument(
        "--input-path",
        default="data/EgoLife/EgoLifeCap/DenseCaption/A1_JAKE",
        help="Input SRT file or directory containing DAY folders.",
    )
    parser.add_argument(
        "--output-path",
        default="data/EgoLife/EgoLifeCap/DenseCaption/translated",
        help="Output JSONL file or directory.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL),
        help=(
            "LLM model name. Uses TRANSLATION_MODEL env var when set, "
            f"otherwise defaults to {DEFAULT_TRANSLATION_MODEL}."
        ),
    )
    args = parser.parse_args()

    print(f"[translate_densecap] Using model: {args.model}")
    translate(args.input_path, args.output_path, args.model)
