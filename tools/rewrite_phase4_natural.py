#!/usr/bin/env python3
"""Rewrite Phase 4 robotic spatial QA into natural Korean/English forms."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TEMPLATE_DESCRIPTIONS = {
    "A": "Where-was-object",
    "B": "What-was-near",
    "C": "Did-I-leave-it-there",
    "D": "Closest-pair",
    "E": "Depth-ordering",
    "F": "Group-locator",
}

WHERE_SEQUENCE = ["A", "B", "C", "E", "F"]
DIST_SEQUENCE = ["D", "B", "E", "C", "F"]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def humanize(value: str) -> str:
    value = str(value).replace("_", " ").strip()
    value = re.sub(r"\s+", " ", value)
    replacements = {
        "I": "나", "Jake": "제이크", "Alice": "앨리스", "Tasha": "타샤", "Katrina": "카트리나", "Lucia": "루시아", "Shure": "슈어",
        "upstairs": "위층", "second floor": "2층", "second-floor": "2층", "living room": "거실", "kitchen": "부엌",
        "ground": "바닥", "floor": "바닥", "table": "테이블", "desk": "책상", "sofa": "소파", "chair": "의자",
        "phone": "휴대폰", "computer": "컴퓨터", "schoolbag": "가방", "food": "음식", "power outlet": "콘센트",
        "whiteboard": "화이트보드", "rack": "랙", "refrigerator": "냉장고", "microwave": "전자레인지", "doorway": "문가",
        "camera": "카메라", "cart": "카트", "milk": "우유", "drinks": "음료", "yogurt": "요거트", "bag": "가방",
        "entryway": "현관", "light": "조명",
        "everyone at the table": "테이블 주변에 있던 모두", "charger": "충전기", "ladle": "국자", "dining table": "식탁",
    }
    lowered = value.lower()
    for src in sorted(replacements, key=len, reverse=True):
        lowered = re.sub(rf"\b{re.escape(src.lower())}\b", replacements[src], lowered)
    return lowered


def pair_parts(text: str) -> tuple[str, str]:
    left, sep, right = text.partition(" and ")
    if not sep:
        return text, "the other item"
    return left.strip(), right.strip()


def triple(question: dict[str, Any]) -> tuple[str, str, str]:
    triples = question.get("evidence_triples") or []
    if triples:
        values = triples[0].get("triple") or ["subject", "related_to", "object"]
        if len(values) >= 3:
            return str(values[0]), str(values[1]), str(values[2])
    return "subject", "related_to", str(question.get("gold_answer", "object"))


def choices_text(question: dict[str, Any]) -> str:
    return ", ".join(f"{key}. {value}" for key, value in question["choices"].items())


def choose_template(question: dict[str, Any], index: int) -> str:
    return (DIST_SEQUENCE if "-DIST-" in question["id"] else WHERE_SEQUENCE)[index % 5]


def rewrite_where(question: dict[str, Any], template_id: str) -> tuple[str, str]:
    subj, pred, obj = triple(question)
    time = question["evidence_chunk_label"]
    subj_kr, obj_kr = humanize(subj), humanize(obj)
    choices = choices_text(question)
    if template_id == "A":
        return (f"{time}쯤 {subj_kr} 어디 있었더라? 보기에서 골라줘.", f"At about {time}, where was {subj}? Choose from the options: {choices}.")
    if template_id == "B":
        return (f"그때 {subj_kr} 근처나 자리로 기억나는 곳이 어디였지? 보기 중에서 골라줘.", f"At that time, what place or nearby anchor was associated with {subj}? Choose from the options: {choices}.")
    if template_id == "C":
        relation = "위에" if pred in {"on", "above"} else "안에" if pred in {"in", "inside"} else "쪽에"
        return (f"{subj_kr}가 {obj_kr} {relation} 있었던 거 맞지? 같은 의미인 답을 보기에서 골라줘.", f"Was {subj} {pred.replace('_', ' ')} {obj}? Pick the option that names the correct place or object: {choices}.")
    if template_id == "E":
        return (f"그 장면에서 {subj_kr} 기준으로 기억해야 할 위치가 어디였어? 보기 중 맞는 걸 골라줘.", f"In that scene, what was the correct remembered location for {subj}? Choose from the options: {choices}.")
    return (f"그 시각에 {subj_kr}가 어느 공간에 속해 있었는지 확인해줘. 보기 중 하나만.", f"At that moment, which place contained {subj}? Choose one option: {choices}.")


def rewrite_dist(question: dict[str, Any], template_id: str) -> tuple[str, str]:
    time = question["evidence_chunk_label"]
    gold = question["gold_answer"]
    first, second = pair_parts(gold)
    first_kr, second_kr = humanize(first), humanize(second)
    choices = choices_text(question)
    if template_id == "D":
        return (f"{time}쯤 서로 제일 가까이 붙어 있던 조합이 뭐였지? 보기에서 골라줘.", f"At about {time}, which subject-object pair was closest? Choose from the options: {choices}.")
    if template_id == "B":
        return (f"그때 {first_kr} 근처에 바로 붙어 있던 건 뭐였어? 가장 가까운 조합을 골라줘.", f"At that time, what was closest to {first}? Choose the closest pair from the options: {choices}.")
    if template_id == "E":
        return (f"앞뒤 깊이까지 보면 어떤 두 대상이 제일 붙어 있었어? 보기 중에서 골라줘.", f"Considering the grounded 3D depth as well, which two targets were closest? Choose from the options: {choices}.")
    if template_id == "C":
        return (f"{first_kr}랑 {second_kr}가 서로 바로 붙어 있던 조합 맞아? 보기에서 맞는 조합을 골라줘.", f"Was the closest pair {first} and {second}? Pick the option with the closest pair: {choices}.")
    return (f"그 시각에 같은 자리처럼 가장 가까웠던 대상 묶음은 뭐였지? 보기 중 하나만.", f"At that moment, which pair occupied the closest shared spot? Choose one option: {choices}.")


def rewrite_question(question: dict[str, Any], index: int, counts: Counter[str]) -> dict[str, Any]:
    template_id = choose_template(question, index)
    counts[template_id] += 1
    kr, en = rewrite_dist(question, template_id) if "-DIST-" in question["id"] else rewrite_where(question, template_id)
    rewritten = dict(question)
    rewritten["original_question"] = question["question"]
    rewritten["question"] = en
    rewritten["question_kr"] = kr
    rewritten["question_en"] = en
    rewritten["template_id"] = template_id
    rewritten["template_description"] = TEMPLATE_DESCRIPTIONS[template_id]
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/golden_qa_phase4.json")
    parser.add_argument("--output", default="output/golden_qa_phase4_natural.json")
    args = parser.parse_args()
    doc = read_json(args.input)
    counts: Counter[str] = Counter()
    questions = [rewrite_question(question, index, counts) for index, question in enumerate(doc["questions"])]
    out = dict(doc)
    out["schema_version"] = 2
    out["source_file"] = args.input
    out["target_balance"] = "phase4 grounded Type1 natural Korean/English conversational rewrite"
    out["template_descriptions"] = TEMPLATE_DESCRIPTIONS
    out["template_counts"] = dict(sorted(counts.items()))
    out["questions"] = questions
    write_json(args.output, out)
    print(f"wrote {args.output}: {len(questions)} questions, templates={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
