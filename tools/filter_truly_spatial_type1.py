#!/usr/bin/env python3
import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

TYPE1_LABEL = "Type1"
SEMANTIC_KEY_RE = re.compile(r'^  "([^"]+)": (\{.*)$')
SRT_FILE_RE = re.compile(r'DAY\d+_(\d{2})(\d{2})(\d{2})\d*\.srt$')
SRT_TIME_RE = re.compile(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})')
LABEL_TIME_RE = re.compile(r'DAY\d+\s+(\d{2}):(\d{2}):(\d{2})')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter Type1 QA rows whose gold answer is not recoverable from non-spatial text."
    )
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--episodic-file", required=True, type=Path)
    parser.add_argument("--semantic-file", required=True, type=Path)
    parser.add_argument("--srt-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def validate_args(args):
    for name in ["sample", "episodic_file", "semantic_file"]:
        path = getattr(args, name)
        if not path.is_file():
            option = name.replace("_", "-")
            raise SystemExit(f"error: --{option} not found: {path}")
    if not args.srt_dir.is_dir():
        raise SystemExit(f"error: --srt-dir not found: {args.srt_dir}")


def normalize_text(value):
    text = str(value).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def singularize_word(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("xes"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ches"):
        return word[:-2]
    if len(word) > 4 and word.endswith("shes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def answer_variants(answer):
    base = normalize_text(answer)
    if not base:
        return []
    variants = {base}
    parts = base.split()
    if parts:
        last_singular = parts[:-1] + [singularize_word(parts[-1])]
        variants.add(" ".join(last_singular))
        variants.add(" ".join(singularize_word(part) for part in parts))
    return sorted(variant for variant in variants if variant)


def contains_answer(text, answer):
    haystack = normalize_text(text)
    if not haystack:
        return False
    return any(variant in haystack for variant in answer_variants(answer))


def triple_text(triples):
    rows = []
    for triple in triples or []:
        if isinstance(triple, (list, tuple)):
            rows.append(" ".join(str(part) for part in triple))
        elif isinstance(triple, dict):
            rows.append(" ".join(str(value) for value in triple.values()))
        else:
            rows.append(str(triple))
    return "\n".join(rows)


def chunk_sort_key(chunk) -> tuple[int, int, str]:
    text = str(chunk)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def sorted_chunks(keys):
    return sorted((str(key) for key in keys), key=chunk_sort_key)


def window_for(keys, center):
    center = str(center)
    if not keys:
        return [center]
    index = {key: pos for pos, key in enumerate(keys)}
    if center in index:
        pos = index[center]
        window = keys[max(0, pos - 1): min(len(keys), pos + 2)]
    else:
        target = chunk_sort_key(center)
        pos = 0
        while pos < len(keys) and chunk_sort_key(keys[pos]) < target:
            pos += 1
        window = keys[max(0, pos - 1): min(len(keys), pos + 2)]
    if center not in window:
        window = list(window) + [center]
    return window


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_semantic_keys(path):
    keys = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = SEMANTIC_KEY_RE.match(line)
            if match:
                keys.append(match.group(1))
    return sorted_chunks(keys)


def load_semantic_target_texts(path, target_keys):
    target_keys = set(str(key) for key in target_keys)
    texts = {key: "" for key in target_keys}
    current_key = None
    collecting = False
    lines = []

    def flush():
        nonlocal current_key, collecting, lines
        if not collecting or current_key is None:
            return
        block = "".join(lines).rstrip()
        if block.endswith(","):
            block = block[:-1].rstrip()
        if block:
            value = json.loads(block)
            texts[current_key] = triple_text(value.get("consolidated_semantic_triples", []))
        collecting = False
        current_key = None
        lines = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = SEMANTIC_KEY_RE.match(line)
            if match:
                flush()
                current_key = match.group(1)
                collecting = current_key in target_keys
                lines = [match.group(2)] if collecting else []
                continue
            if collecting:
                if line == line.lstrip() and line.strip() == "}":
                    flush()
                    break
                lines.append(line)
    flush()
    return texts


def parse_srt_time(value):
    match = SRT_TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_label_seconds(label):
    match = LABEL_TIME_RE.fullmatch(str(label).strip())
    if not match:
        raise ValueError(f"Invalid evidence_chunk_label: {label}")
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def file_base_seconds(path):
    match = SRT_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer SRT base time from {path.name}")
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def parse_srt_file(path):
    base = file_base_seconds(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    segments = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        start_text, end_text = [part.strip() for part in lines[time_index].split("-->", 1)]
        body = " ".join(lines[time_index + 1:])
        if not body:
            continue
        segments.append(
            {
                "start": base + parse_srt_time(start_text),
                "end": base + parse_srt_time(end_text),
                "text": body,
            }
        )
    return segments


def load_srt_segments(root):
    if not root.exists():
        return []
    segments = []
    for path in sorted(root.glob("*.srt")):
        segments.extend(parse_srt_file(path))
    return segments


def srt_text_for_window(segments, center_seconds, half_window=30):
    start = center_seconds - half_window
    end = center_seconds + half_window
    return "\n".join(
        segment["text"]
        for segment in segments
        if segment["end"] >= start and segment["start"] <= end
    )


def escape_md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def bool_word(value):
    return "yes" if value else "no"


def evidence_triple(question):
    triples = question.get("evidence_triples") or []
    if triples and isinstance(triples[0], dict):
        triple = triples[0].get("triple")
        if isinstance(triple, list) and len(triple) >= 3:
            return triple
    return None


def pattern_summary(questions):
    if not questions:
        return "none"
    subjects = Counter()
    predicates = Counter()
    subject_predicates = Counter()
    for question in questions:
        triple = evidence_triple(question)
        if not triple:
            continue
        subject = normalize_text(triple[0]) or str(triple[0])
        predicate = normalize_text(triple[1]) or str(triple[1])
        subjects[subject] += 1
        predicates[predicate] += 1
        subject_predicates[f"{subject}/{predicate}"] += 1
    return (
        f"subjects {format_counter(subjects)}; "
        f"predicates {format_counter(predicates)}; "
        f"subject/predicate {format_counter(subject_predicates)}"
    )


def format_counter(counter, limit=5):
    if not counter:
        return "none"
    return ", ".join(f"{key} ({count})" for key, count in counter.most_common(limit))


def write_report(path, audit_rows, true_questions, leaked_questions):
    total = len(audit_rows)
    true_count = len(true_questions)
    leaked_count = len(leaked_questions)
    kept_ids = ", ".join(question["id"] for question in true_questions) or "none"
    leaked_ids = ", ".join(question["id"] for question in leaked_questions) or "none"
    leaked_fraction = leaked_count / total if total else 0
    kept_fraction = true_count / total if total else 0

    lines = [
        "# Type 1 Spatial-Required Filter",
        "",
        "## Methodology",
        "- Audited only rows with `type_label == Type1` from `output/golden_qa_sample.json`.",
        "- For episodic memory, checked the `evidence_chunk` plus one previous and one next episodic chunk.",
        "- For semantic memory, streamed the 131 MB consolidated file and checked the same +/- one chunk window in semantic chunk order.",
        "- For DenseCaption and Transcript SRT, checked segments overlapping `evidence_chunk_label` +/- 30 seconds.",
        "- Matching was deterministic: case-insensitive normalized substring plus basic singularization of the gold answer. No LLM and no synonym expansion beyond text that appears verbatim.",
        "",
        "## Per-Q Audit",
        "| id | gold_answer | found-in-episodic | found-in-semantic | found-in-densecaption | found-in-transcript | truly_spatial_required |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in audit_rows:
        check = row["evidence_check"]
        lines.append(
            "| {id} | {answer} | {episodic} | {semantic} | {dense} | {transcript} | {true} |".format(
                id=escape_md(row["id"]),
                answer=escape_md(row["gold_answer"]),
                episodic=bool_word(check["episodic_found"]),
                semantic=bool_word(check["semantic_found"]),
                dense=bool_word(check["dense_caption_found"]),
                transcript=bool_word(check["transcript_found"]),
                true=bool_word(row["truly_spatial_required"]),
            )
        )

    lines.extend(
        [
            "",
            "## Summary",
            f"- Truly spatial-required: {true_count}/{total} ({kept_fraction:.1%}). Leaked/recoverable from non-spatial text: {leaked_count}/{total} ({leaked_fraction:.1%}).",
            f"- Truly spatial-required dominant patterns: {pattern_summary(true_questions)}.",
            f"- Leaked/recoverable dominant patterns: {pattern_summary(leaked_questions)}.",
            "",
            "## Recommendation",
            f"- Keep for the deck: {kept_ids}.",
            f"- Drop or relabel before claiming a spatial-axis-only effect: {leaked_ids}.",
            f"- Re-running ablation on this subset would change the Type1 pool by removing {leaked_count}/{total} ({leaked_fraction:.1%}) cases and keeping {true_count}/{total} ({kept_fraction:.1%}). The Phase 3b -4.2pp result should be treated as diluted by leaked Type1 cases until the subset is re-run; direction or magnitude cannot be derived from filtering alone.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_counts(output_data):
    questions = output_data.get("questions", [])
    if isinstance(output_data.get("counts"), dict):
        labels = list(output_data["counts"].keys())
        counts = Counter(question.get("type_label") for question in questions)
        output_data["counts"] = {label: counts.get(label, 0) for label in labels}
    if "target_balance" in output_data:
        counts = output_data.get("counts", {})
        output_data["target_balance"] = "-".join(str(counts.get(label, 0)) for label in ["Type1", "Type2", "Type3", "Type4"])


def classify(args):
    sample = load_json(args.sample)
    type1_questions = [question for question in sample.get("questions", []) if question.get("type_label") == TYPE1_LABEL]

    episodic_data = load_json(args.episodic_file)
    episodic_triples = episodic_data.get("episodic_triples", {})
    episodic_keys = sorted_chunks(episodic_triples.keys())
    episodic_texts = {str(key): triple_text(value) for key, value in episodic_triples.items()}

    semantic_keys = collect_semantic_keys(args.semantic_file)
    semantic_targets = set()
    for question in type1_questions:
        semantic_targets.update(window_for(semantic_keys, question.get("evidence_chunk", "")))
    semantic_texts = load_semantic_target_texts(args.semantic_file, semantic_targets)

    dense_segments = load_srt_segments(args.srt_dir / "DenseCaption" / "A1_JAKE" / "DAY1")
    transcript_segments = load_srt_segments(args.srt_dir / "Transcript" / "A1_JAKE" / "DAY1")

    audit_rows = []
    filtered_questions = []
    leaked_questions = []

    for question in type1_questions:
        chunk = str(question.get("evidence_chunk", ""))
        label = question.get("evidence_chunk_label", "")
        answer = question.get("gold_answer", "")
        episodic_window = window_for(episodic_keys, chunk)
        semantic_window = window_for(semantic_keys, chunk)
        center_seconds = parse_label_seconds(label)

        episodic_text = "\n".join(episodic_texts.get(key, "") for key in episodic_window)
        semantic_text = "\n".join(semantic_texts.get(key, "") for key in semantic_window)
        dense_text = srt_text_for_window(dense_segments, center_seconds)
        transcript_text = srt_text_for_window(transcript_segments, center_seconds)

        evidence_check = {
            "episodic_found": contains_answer(episodic_text, answer),
            "semantic_found": contains_answer(semantic_text, answer),
            "dense_caption_found": contains_answer(dense_text, answer),
            "transcript_found": contains_answer(transcript_text, answer),
        }
        truly_spatial_required = not any(evidence_check.values())

        audited_question = copy.deepcopy(question)
        audited_question["truly_spatial_required"] = truly_spatial_required
        audited_question["evidence_check"] = evidence_check
        audit_rows.append(
            {
                "id": question.get("id", ""),
                "gold_answer": answer,
                "evidence_check": evidence_check,
                "truly_spatial_required": truly_spatial_required,
                "question": audited_question,
            }
        )
        if truly_spatial_required:
            filtered_questions.append(audited_question)
        else:
            leaked_questions.append(audited_question)

    output_data = copy.deepcopy(sample)
    output_data["questions"] = filtered_questions
    update_counts(output_data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, audit_rows, filtered_questions, leaked_questions)

    total = len(type1_questions)
    true_count = len(filtered_questions)
    leaked_count = total - true_count
    leaked_fraction = leaked_count / total if total else 0
    print(
        f"Type 1 spatial-required filter: {true_count}/{total} remain truly spatial-required; "
        f"{leaked_count}/{total} ({leaked_fraction:.1%}) are recoverable from episodic/semantic/caption/transcript text. "
        f"Implication for Phase 3b -4.2pp: the finding mixes true spatial cases with leaked Type1 rows, so it is likely diluted; rerun on the filtered subset is needed before interpreting the spatial-axis effect size."
    )


def main():
    args = parse_args()
    validate_args(args)
    classify(args)


if __name__ == "__main__":
    main()
