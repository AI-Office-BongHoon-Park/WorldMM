import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pysrt
from tqdm import tqdm

from worldmm.llm import LLMModel

_MODEL_NAME = os.environ.get("WORLDMM_LLM_MODEL", "gpt-5-mini")
model = LLMModel(model_name=_MODEL_NAME)

SYSTEM_PROMPT = "You are a helpful assistant that translates text from Chinese to English. Answer in translated text only."


def build_record(name: str, date: str, start: str, end: str, idx: int, translation: str) -> dict:
    return {
        "custom_id": f"{idx}-{name}-{date}-{start}-{end}",
        "translated_text": translation
    }


def translate(input_path: str, output_path: str) -> None:
    """
    Translate a single SRT file or every SRT file under a directory (e.g., DenseCaption/A1_JAKE/DAY*).
    - input_path: path to an SRT file or a directory containing DAY folders.
    - output_path: output file if input_path is a file, otherwise output directory for all translated JSONLs.
    """
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
        if day_dirs:
            for day in day_dirs:
                day_path = os.path.join(input_path, day)
                if not os.path.isdir(day_path):
                    continue
                files = [f for f in os.listdir(day_path) if f.endswith(".srt")]
                for file in files:
                    input_file = os.path.join(day_path, file)
                    output_file = os.path.join(output_path, file.replace(".srt", ".jsonl"))
                    file_pairs.append((input_file, output_file))
        else:
            files = [f for f in os.listdir(input_path) if f.endswith(".srt")]
            for file in files:
                input_file = os.path.join(input_path, file)
                output_file = os.path.join(output_path, file.replace(".srt", ".jsonl"))
                file_pairs.append((input_file, output_file))

        file_pairs = [(i, o) for i, o in file_pairs if not os.path.exists(o)]
        if not file_pairs:
            print("All target SRT files already translated; nothing to do.")
            return
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_translate_file, in_f, out_f): in_f for in_f, out_f in file_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating files"):
                future.result()
    else:
        _translate_file(input_path, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/EgoLife/EgoLifeCap/DenseCaption/A1_JAKE",
                        help="SRT file or directory of DAY folders.")
    parser.add_argument("--output", default="data/EgoLife/EgoLifeCap/DenseCaption/translated",
                        help="Output JSONL file (if input is a file) or directory.")
    parser.add_argument("--day", default=None,
                        help="If input is a directory, restrict to a single DAY (e.g. DAY1).")
    args = parser.parse_args()

    input_path = args.input
    if args.day and os.path.isdir(input_path):
        input_path = os.path.join(input_path, args.day)

    translate(input_path, args.output)
