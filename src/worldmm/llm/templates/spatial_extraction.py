"""
Template for extracting spatial knowledge (WHERE-axis) from episodic triples.
"""

spatial_extraction_system = """You are tasked with extracting spatial knowledge from episodic triples.
Your goal is to capture facts about WHERE things are: object/agent locations, the room or place they occupy, and how objects are positioned relative to each other.

# What to Extract:
1. **Containment / Placement**: an object or person is at or inside a place
   (e.g., ["Alice", "located_in", "kitchen"], ["book", "on", "coffee table"]).
2. **Relative position between concrete objects/agents**: which thing is next to / behind / in front of which
   (e.g., ["Alice", "next_to", "Bob"], ["sofa", "left_of", "tv stand"]).
3. **Place composition**: a fixed object that is a permanent part of a place
   (e.g., ["living room", "contains", "sofa"]).

# Predicate Vocabulary (CLOSED SET — use exactly one of these, lowercase, snake_case):
in, on, under, next_to, near, left_of, right_of, behind, in_front_of, contains, located_in

# What to Avoid:
- **Pure event/action triples without a spatial cue** (e.g., ["Alice", "talks to", "Bob"]).
- **Predicates outside the closed set above.** If the relation doesn't fit, drop the triple.
- **Speculative location inferences** not supported by the episodic input.
- **Repeated identical (subject, predicate, object) triples within the same chunk.** Keep one.

# Important Notes:
- Each spatial triple MUST be grounded in at least one supporting episodic triple.
- Place names should be normalized lowercase, snake_case if multi-word (e.g., "living_room", "balcony").
- Person and object names should preserve their original casing.
- If multiple episodic triples support the same spatial fact, merge them into one triple and list all supporting indices in episodic_evidence.

# Output Format:
Return ONLY a JSON object with these two keys:
- `spatial_triples` (List[List[str]]): Each item is a triple [subject, predicate, object] with predicate from the closed vocab.
- `episodic_evidence` (List[List[int]]): Each item is a list of **0-based** indices into the input episodic triples that support the corresponding spatial triple at the same position.
- Both lists MUST have the same length and aligned order.
- If no spatial knowledge is inferable, return:
  {"spatial_triples": [], "episodic_evidence": []}
"""

one_shot_spatial_input = """Episodic triples:
0. ["Jake", "sits on", "sofa"],
1. ["Jake", "places", "book on the coffee table"],
2. ["Maria", "sits beside", "Jake"],
3. ["sofa", "is in", "the living room"],
4. ["coffee table", "stands in front of", "sofa"],
5. ["Jake", "talks to", "Maria"]
"""

one_shot_spatial_output = """{
  "spatial_triples": [
    ["Jake", "on", "sofa"],
    ["book", "on", "coffee_table"],
    ["Maria", "next_to", "Jake"],
    ["sofa", "located_in", "living_room"],
    ["coffee_table", "in_front_of", "sofa"]
  ],
  "episodic_evidence": [
    [0],
    [1],
    [2],
    [3],
    [4]
  ]
}
"""


prompt_template = [
    {"role": "system", "content": spatial_extraction_system},
    {"role": "user", "content": one_shot_spatial_input},
    {"role": "assistant", "content": one_shot_spatial_output},
    {"role": "user", "content": "Episodic triples:\n${episodic_triples}"}
]
