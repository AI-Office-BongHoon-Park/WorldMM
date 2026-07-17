"""
Template for consolidating a single spatial triple against relevant existing ones.
Closed predicate vocab MUST be respected.
"""

spatial_consolidation_system = """You consolidate spatial knowledge. You are given a NEW spatial triple and a list of EXISTING spatial triples from prior timestamps that may overlap with it.

You must decide two things:
1. **Which existing triples to remove/pop** — those that say essentially the same thing as the new triple, or that conflict with it (e.g., an object can't be in two different rooms at the same time).
2. **How to update the new triple** — to capture the most accurate/specific spatial fact after merging.

# Consolidation Rules:
1. **Merge near-duplicates** by removing the older redundant existing triple and keeping the new triple (possibly tightened).
2. **Resolve location conflicts** (e.g., ["X","located_in","kitchen"] vs ["X","located_in","living_room"]): assume the new triple is more recent and remove the conflicting existing one.
3. **Preserve unique spatial facts.** Do not remove existing triples that say something independent.
4. **The updated predicate MUST be from the closed vocab**:
   in, on, under, next_to, near, left_of, right_of, behind, in_front_of, contains, located_in.

# Output Format:
Return ONLY a JSON object with:
- `updated_triple` (List[str]): the new triple, possibly refined, as [subject, predicate, object].
- `triples_to_remove` (List[int]): 0-based indices into the existing triples list; empty if none.
"""

one_shot_consolidation_input = """New triple:
["Jake", "located_in", "living_room"]

Existing triples:
0. ["Jake", "located_in", "kitchen"]
1. ["Jake", "next_to", "Maria"]
2. ["Jake", "in", "living_room"]
"""

one_shot_consolidation_output = """{
  "updated_triple": ["Jake", "located_in", "living_room"],
  "triples_to_remove": [0, 2]
}"""


prompt_template = [
    {"role": "system", "content": spatial_consolidation_system},
    {"role": "user", "content": one_shot_consolidation_input},
    {"role": "assistant", "content": one_shot_consolidation_output},
    {"role": "user", "content": "New triple:\n${new_triple}\n\nExisting triples:\n${existing_triples}"}
]
