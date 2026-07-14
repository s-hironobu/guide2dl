# gpt1_physics_qaator.py - Physics concept QA dataset generator for 10-year-old level
#
# Purpose: Learn fundamental concepts of physics using a 10-year-old level vocabulary (24 labels, large vocabulary).
#
#          Out of a massive combinatorial space (entity \times sentence pattern \times question phrasing \times emphasis word),
#           we use only a tiny fraction for training.
#           We then test if the model can correctly answer:
#           (A) Words not seen during training (unseen words).
#           (B) Words seen during training but combined with an unseen sentence pattern (unseen combinations).
#           This verifies whether the model has truly learned the underlying concepts rather than just memorizing lines.
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import os
import sys
import random
import itertools

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "text_processing"))

from text_normalizer import fix_text


# ============================================================
# Concept Definitions (12 concepts \times 2 polarities = 24 labels)
# ============================================================

_ADVERBS = ["very", "really", "extremely", "incredibly", ""]  # "" = no emphasis word

# Introductory phrases shared across all concepts.
# Prepending these to the scenario sentence expands the combinatorial space uniformly
# without needing to manually increase the number of individual templates.
_STARTERS = [
    "",
    "Imagine this: ",
    "Here is what happens: ",
    "Picture this scene: ",
    "In this situation, ",
]


def _apply_starter(text: str, starter: str) -> str:
    """Prepends an introductory phrase. If the starter is empty, it just capitalizes the first letter of the text."""
    if not starter:
        return text
    # Since the starter ends a sentence clause, convert the first letter of the main text to lowercase for natural flow.
    return starter + text[0].lower() + text[1:]


def _mk(
    pos_label,
    neg_label,
    pos_entities,
    neg_entities,
    pos_templates,
    neg_templates,
    questions,
):
    return {
        "pos_label": pos_label,
        "neg_label": neg_label,
        "pos_entities": pos_entities,
        "neg_entities": neg_entities,
        "pos_templates": pos_templates,
        "neg_templates": neg_templates,
        "questions": questions,
    }


CONCEPTS = {
    "speed": _mk(
        "fast",
        "slow",
        pos_entities=[
            "cheetah",
            "rabbit",
            "falcon",
            "race car",
            "sports car",
            "jet plane",
            "race horse",
            "dolphin",
            "bullet train",
            "greyhound",
            "hawk",
            "motorcycle",
            "speedboat",
            "eagle",
        ],
        neg_entities=[
            "turtle",
            "snail",
            "sloth",
            "tortoise",
            "garden slug",
            "old tractor",
            "glacier",
            "tired old dog",
            "inchworm",
            "ox pulling a cart",
            "sleepy cat",
            "koala",
            "starfish",
            "hour hand of a clock",
        ],
        pos_templates=[
            "A {e} runs {adv} quickly.",
            "The {e} moves at a {adv} high speed.",
            "Watch the {e} zoom by in a flash.",
            "The {e} dashes across the field in seconds.",
            "The {e} speeds past everyone {adv} fast.",
        ],
        neg_templates=[
            "A {e} moves {adv} slowly.",
            "The {e} barely moves at all.",
            "The {e} takes a {adv} long time to go anywhere.",
            "The {e} creeps along at a {adv} slow pace.",
            "The {e} inches forward {adv} slowly.",
        ],
        questions=[
            "Is it fast or slow?",
            "How does it move, fast or slow?",
            "Would you say it is fast or slow?",
            "Does it move quickly or slowly?",
        ],
    ),
    "gravity": _mk(
        "falls",
        "stays",
        pos_entities=[
            "apple",
            "pencil",
            "coin",
            "leaf",
            "book",
            "raindrop",
            "acorn",
            "toy",
            "ball",
            "key",
            "marble",
            "feather",
        ],
        neg_entities=[
            "book on a shelf",
            "cup on a table",
            "chair on the floor",
            "picture hung on a wall",
            "vase on a shelf",
            "lamp on a desk",
            "plate on a table",
            "rug on the floor",
            "clock on the wall",
            "box on the ground",
        ],
        pos_templates=[
            "You let go of the {e} in the air.",
            "You drop the {e} from your hand.",
            "The {e} slips out of your fingers above the floor.",
            "You release the {e} while holding it up high.",
        ],
        neg_templates=[
            "You gently place the {e}.",
            "You set the {e} down carefully.",
            "The {e} is resting there, {adv} steady.",
            "Someone put the {e} there {adv} carefully.",
        ],
        questions=[
            "What happens to it?",
            "What happens next?",
            "What will it do?",
            "What happens after that?",
        ],
    ),
    "friction": _mk(
        "slides",
        "grips",
        pos_entities=[
            "ice",
            "wet tile floor",
            "banana peel",
            "frozen pond",
            "polished marble floor",
            "wet grass in the rain",
            "oily garage floor",
            "waxed wooden floor",
        ],
        neg_entities=[
            "dry sand",
            "rubber mat",
            "carpet",
            "dry pavement",
            "rough tree bark",
            "gravel path",
            "sandpaper",
            "rocky trail",
        ],
        pos_templates=[
            "You walk on the {e}.",
            "You step onto the {e}.",
            "You try walking across the {e}.",
            "Your shoes touch the {e}.",
        ],
        neg_templates=[
            "You walk on the {e}.",
            "You step onto the {e}.",
            "You try walking across the {e}.",
            "Your shoes touch the {e}.",
        ],
        questions=[
            "What happens when you walk on it?",
            "What happens to your feet?",
            "Do you slide or not?",
            "Is it slippery or not?",
        ],
    ),
    "buoyancy": _mk(
        "floats",
        "sinks",
        pos_entities=[
            "wooden block",
            "rubber duck",
            "empty plastic bottle",
            "beach ball",
            "cork",
            "leaf",
            "piece of styrofoam",
            "balloon filled with air",
            "small boat",
            "empty water bottle",
        ],
        neg_entities=[
            "steel coin",
            "rock",
            "brick",
            "iron nail",
            "heavy stone",
            "metal spoon",
            "glass marble",
            "anchor",
            "bowling ball",
            "can filled with sand",
        ],
        pos_templates=[
            "You put the {e} in the water.",
            "You drop the {e} into the pool.",
            "The {e} is placed on the surface of the lake.",
            "You gently set the {e} on the pond.",
        ],
        neg_templates=[
            "You put the {e} in the water.",
            "You drop the {e} into the pool.",
            "The {e} is placed in the lake.",
            "You gently release the {e} into the pond.",
        ],
        questions=[
            "Does it float or sink?",
            "What happens to it in the water?",
            "Will it float on top or go under?",
        ],
    ),
    "hardness": _mk(
        "hard",
        "soft",
        pos_entities=[
            "rock",
            "diamond",
            "steel bar",
            "brick",
            "wooden table",
            "hammer head",
            "marble statue",
            "concrete block",
            "metal pipe",
            "ceramic plate",
        ],
        neg_entities=[
            "pillow",
            "marshmallow",
            "cotton ball",
            "sponge",
            "piece of foam",
            "soft blanket",
            "plush teddy bear",
            "bread dough",
            "beanbag",
            "piece of jelly",
        ],
        pos_templates=[
            "You touch the {e}.",
            "You press your hand on the {e}.",
            "You squeeze the {e} {adv} hard.",
            "You tap the {e} with your finger.",
        ],
        neg_templates=[
            "You touch the {e}.",
            "You press your hand on the {e}.",
            "You squeeze the {e} {adv} gently.",
            "You poke the {e} with your finger.",
        ],
        questions=[
            "Does it feel hard or soft?",
            "Is it hard or soft?",
            "How does it feel when you touch it?",
        ],
    ),
    "elasticity": _mk(
        "bounces",
        "no_bounce",
        pos_entities=[
            "rubber ball",
            "basketball",
            "tennis ball",
            "bouncy ball",
            "soccer ball",
            "spring",
            "pogo stick",
            "stretched rubber band",
        ],
        neg_entities=[
            "glass cup",
            "egg",
            "water balloon",
            "clay ball",
            "wet sponge",
            "bag of flour",
            "raw tomato",
            "paper ball",
        ],
        pos_templates=[
            "You drop the {e} on the floor.",
            "The {e} hits the ground {adv} hard.",
            "You throw the {e} down onto the pavement.",
        ],
        neg_templates=[
            "You drop the {e} on the floor.",
            "The {e} hits the ground {adv} hard.",
            "You throw the {e} down onto the pavement.",
        ],
        questions=[
            "Does it bounce back up?",
            "What happens when it hits the ground?",
            "Does it bounce or not?",
        ],
    ),
    "magnetism": _mk(
        "attracts",
        "no_attract",
        pos_entities=[
            "iron nail",
            "steel paperclip",
            "tin can",
            "steel spoon",
            "refrigerator door",
            "steel screw",
            "bobby pin",
            "metal staple",
            "steel washer",
            "metal fork",
        ],
        neg_entities=[
            "wooden pencil",
            "plastic ruler",
            "glass marble",
            "rubber eraser",
            "cotton shirt",
            "piece of paper",
            "aluminum foil ball",
            "copper coin",
            "ceramic mug",
        ],
        pos_templates=[
            "You bring a magnet near the {e}.",
            "You hold a magnet close to the {e}.",
            "A magnet is placed next to the {e}.",
        ],
        neg_templates=[
            "You bring a magnet near the {e}.",
            "You hold a magnet close to the {e}.",
            "A magnet is placed next to the {e}.",
        ],
        questions=[
            "Does the magnet pull it?",
            "What happens between the magnet and it?",
            "Is it attracted to the magnet or not?",
        ],
    ),
    "sound": _mk(
        "loud",
        "quiet",
        pos_entities=[
            "fire truck siren",
            "thunderclap",
            "rock concert",
            "jet engine",
            "barking large dog",
            "drum being hit hard",
            "ambulance siren",
            "marching band",
            "jackhammer",
            "fireworks",
        ],
        neg_entities=[
            "whisper",
            "falling feather",
            "ticking wristwatch",
            "cat purring",
            "gentle breeze",
            "candle flame flickering",
            "falling snow",
            "butterfly flying",
            "soft lullaby",
            "mouse tiptoeing",
        ],
        pos_templates=[
            "You hear a {e}.",
            "A {e} suddenly happens nearby.",
            "The sound of a {e} fills the air.",
        ],
        neg_templates=[
            "You hear a {e}.",
            "A {e} happens nearby.",
            "The sound of a {e} is in the air.",
        ],
        questions=[
            "Is it loud or quiet?",
            "How does it sound, loud or quiet?",
            "Would you say it is loud or quiet?",
        ],
    ),
    "light": _mk(
        "bright",
        "dark",
        pos_entities=[
            "the sun",
            "a spotlight",
            "a lightning flash",
            "a flashlight beam",
            "a lighthouse beacon",
            "fireworks in the sky",
            "car headlights",
            "a bright lamp",
            "a camera flash",
        ],
        neg_entities=[
            "a cave with no light",
            "a moonless night sky",
            "a closed closet",
            "a basement with no windows",
            "a room with the lights off",
            "a forest at midnight",
            "a tunnel with no lights",
            "a total eclipse",
        ],
        pos_templates=[
            "You look at {e}.",
            "You see {e} nearby.",
            "{e} shines in front of you.",
        ],
        neg_templates=[
            "You are inside {e}.",
            "You look around {e}.",
            "You find yourself in {e}.",
        ],
        questions=[
            "Is it bright or dark?",
            "Can you see well, or is it hard to see?",
            "Would you say it is bright or dark?",
        ],
    ),
    "temperature": _mk(
        "hot",
        "cold",
        pos_entities=[
            "stove burner",
            "campfire",
            "cup of fresh coffee",
            "summer sun",
            "heater",
            "oven",
            "boiling water",
            "candle flame",
            "hot iron",
            "pavement in summer",
        ],
        neg_entities=[
            "ice cube",
            "freezer",
            "snowman",
            "glass of ice water",
            "winter morning",
            "popsicle",
            "snowy mountain peak",
            "refrigerator",
            "cold winter wind",
            "frozen lake",
        ],
        pos_templates=[
            "You touch the {e}.",
            "You get close to the {e}.",
            "You feel the {e} nearby.",
        ],
        neg_templates=[
            "You touch the {e}.",
            "You get close to the {e}.",
            "You feel the {e} nearby.",
        ],
        questions=[
            "Does it feel hot or cold?",
            "Is it hot or cold?",
            "How does it feel?",
        ],
    ),
    "rolling": _mk(
        "rolls",
        "no_roll",
        pos_entities=[
            "soccer ball",
            "basketball",
            "marble",
            "wheel",
            "orange",
            "can lying on its side",
            "bowling ball",
            "tennis ball",
            "globe",
        ],
        neg_entities=[
            "cube-shaped box",
            "book",
            "brick",
            "chair",
            "table",
            "picture frame",
            "shoe box",
            "triangular block",
            "stack of paper",
        ],
        pos_templates=[
            "You give the {e} a small push.",
            "You nudge the {e} on the floor.",
            "The {e} is pushed gently across the ground.",
        ],
        neg_templates=[
            "You give the {e} a small push.",
            "You nudge the {e} on the floor.",
            "The {e} is pushed gently across the ground.",
        ],
        questions=[
            "Does it roll away?",
            "What happens when you push it?",
            "Does it roll or stay put?",
        ],
    ),
    "breakability": _mk(
        "breaks",
        "unbreakable",
        pos_entities=[
            "glass cup",
            "egg",
            "ceramic plate",
            "light bulb",
            "thin cookie",
            "porcelain vase",
            "glass jar",
            "china teacup",
            "clay pot",
            "mirror",
        ],
        neg_entities=[
            "rubber ball",
            "plastic cup",
            "metal spoon",
            "wooden block",
            "steel water bottle",
            "tennis ball",
            "rock",
            "leather shoe",
            "plastic toy car",
            "metal key",
        ],
        pos_templates=[
            "You drop the {e} on the hard floor.",
            "The {e} falls onto the concrete {adv} hard.",
            "You accidentally knock the {e} off the table.",
        ],
        neg_templates=[
            "You drop the {e} on the hard floor.",
            "The {e} falls onto the concrete {adv} hard.",
            "You accidentally knock the {e} off the table.",
        ],
        questions=[
            "What happens to it?",
            "Does it break?",
            "Is it broken or still fine?",
        ],
    ),
}


# ============================================================
# Build label list
# ============================================================

PHYSICS_LABELS: list[str] = []
for _c in CONCEPTS.values():
    PHYSICS_LABELS.append(_c["pos_label"])
    PHYSICS_LABELS.append(_c["neg_label"])

assert len(PHYSICS_LABELS) == len(
    set(PHYSICS_LABELS)
), "Duplicate labels found in PHYSICS_LABELS"

LABEL_TO_ID = {lbl: i for i, lbl in enumerate(PHYSICS_LABELS)}
ID_TO_LABEL = {i: lbl for i, lbl in enumerate(PHYSICS_LABELS)}
NUM_PHYSICS_LABELS = len(PHYSICS_LABELS)


# ============================================================
# Split entities for train/val (unseen entity holdout)
# ============================================================

_ENTITY_HOLDOUT_RATIO = (
    0.25  # Hold out 25% of entities per polarity as val-only unseen words
)


def _split_entities(
    entities: list[str], rng: random.Random
) -> tuple[list[str], list[str]]:
    """Randomly split entities into (train, val-only unseen holdouts)."""
    shuffled = entities[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * _ENTITY_HOLDOUT_RATIO))
    holdout = shuffled[:n_holdout]
    train = shuffled[n_holdout:]
    return train, holdout


# ============================================================
# Generate combinations (entity * template * question * adverb)
# ============================================================


def _expand_combos(entities, templates, questions) -> list[tuple[str, str, str]]:
    """
    Expand all combinations of (entity, template, question, starter).
    If the template contains {adv}, expand with variation of adverbs.
    Additionally, expand with introductory phrases (_STARTERS) common
    to all concepts to uniformly expand the combination space.

    Returns
    -------
    list of (scenario_text, question_text, entity) - entity is used for deduplication.
    """
    combos = []
    for entity, tpl, q in itertools.product(entities, templates, questions):
        if "{adv}" in tpl:
            base_texts = []
            for adv in _ADVERBS:
                text = tpl.format(e=entity, adv=adv)
                text = " ".join(
                    text.split()
                )  # Normalize consecutive spaces to a single space.
                base_texts.append(text)
        else:
            base_texts = [tpl.format(e=entity)]

        for base_text in base_texts:
            for starter in _STARTERS:
                final_text = _apply_starter(base_text, starter)
                combos.append((fix_text(final_text), fix_text(q), entity))
    return combos


# ============================================================
# Compound scenarios with 2 concepts (Designed to force question dependency)
# ============================================================
#
# Problem: In single-concept scenarios, the correct answer is always the same
#          for that scenario regardless of the question wording. This allows
#          the model to answer correctly using only the scenario, without
#          reading the question at all (question-independent bias).
#
# Solution: Mix two different concept entities into a single sentence, so that
#           the answer changes depending on "which entity is being asked about".
#           The question must explicitly include the target entity name.
#
#   Example: "The cheetah runs very quickly, and the rock feels very hard."
#            Q: "Is the cheetah fast or slow?"  -> fast
#            Q: "Is the rock hard or soft?"     -> hard
#
# By doing this, multiple (question, correct_label) pairs will exist for the
# same scenario string, making it impossible to answer correctly without
# reading the question.


_COMPOUND_JOINERS = [
    ", and ",
    ".  Meanwhile, ",
    ".  At the same time, ",
    ";  also, ",
]


def _entity_question(question_tpl: str, entity: str) -> str:
    """
    Create a question by replacing the pronoun "it" in a generic template
    (e.g., "Is it fast or slow?") with a specific entity name (e.g., "Is the cheetah fast or slow?").
    This is handled by simply replacing "it" / "it," with the entity name.
    """
    # "Is it X or Y?" -> "Is the {entity} X or Y?"
    replaced = question_tpl.replace(" it ", f" the {entity} ")
    replaced = replaced.replace("Is it", f"Is the {entity}")
    replaced = replaced.replace("Does it", f"Does the {entity}")
    replaced = replaced.replace("Would you say it is", f"Would you say the {entity} is")
    return replaced


def _sample_one(spec: dict, polarity: str, entities: list[str], rng: random.Random):
    """Randomly select one entity, one template, and one question from the specified polarity side of the spec."""
    entity = rng.choice(entities)
    tpl = rng.choice(spec[f"{polarity}_templates"])
    q = rng.choice(spec["questions"])
    if "{adv}" in tpl:
        adv = rng.choice(_ADVERBS)
        text = " ".join(tpl.format(e=entity, adv=adv).split())
    else:
        text = tpl.format(e=entity)
    label_id = LABEL_TO_ID[spec[f"{polarity}_label"]]
    return entity, text, q, label_id


def _generate_compound_examples(
    n_examples: int,
    rng: random.Random,
    concept_entity_pools: dict,
) -> list[dict]:
    """
    Generate `n_examples` of compound scenarios combining two different concepts.
    From a single compound scenario, two independent (question, label) records
    are generated --- one for entity A and one for entity B
    (meaning even if the scenario is the same, different questions yield different answers).

    concept_entity_pools: {concept_name: {"pos": (train_entities, val_entities), "neg": (...)}}
                           The caller passes whether to use the entity pool for
                           training or validation.
    """
    concept_names = list(concept_entity_pools.keys())
    records: list[dict] = []

    n_pairs = max(1, n_examples // 2)  # Since one pair generates two records

    for _ in range(n_pairs):
        c1, c2 = rng.sample(concept_names, 2)
        spec1, spec2 = CONCEPTS[c1], CONCEPTS[c2]
        pol1 = rng.choice(["pos", "neg"])
        pol2 = rng.choice(["pos", "neg"])
        entities1 = concept_entity_pools[c1][pol1]
        entities2 = concept_entity_pools[c2][pol2]
        if not entities1 or not entities2:
            continue

        entity1, frag1, q1_tpl, label1 = _sample_one(spec1, pol1, entities1, rng)
        entity2, frag2, q2_tpl, label2 = _sample_one(spec2, pol2, entities2, rng)

        joiner = rng.choice(_COMPOUND_JOINERS)
        # Lowercase the start of the second fragment for a natural connection
        frag2_lower = frag2[0].lower() + frag2[1:]
        combined_scenario = fix_text(frag1.rstrip(".") + joiner + frag2_lower)

        starter = rng.choice(_STARTERS)
        combined_scenario = _apply_starter(combined_scenario, starter)

        q1 = fix_text(_entity_question(q1_tpl, entity1))
        q2 = fix_text(_entity_question(q2_tpl, entity2))

        records.append({"scenario": combined_scenario, "question": q1, "label": label1})
        records.append({"scenario": combined_scenario, "question": q2, "label": label2})

    return records


# ============================================================
# Core Dataset Generation
# ============================================================

DEFAULT_TRAIN_SAMPLES_PER_CONCEPT = (
    160  # Number of train samples per concept (sum of both polarities)
)
DEFAULT_VAL_ENTITY_SAMPLES_PER_CONCEPT = (
    30  # Number of unseen-word validation samples per concept
)
DEFAULT_VAL_COMBO_SAMPLES_PER_CONCEPT = (
    30  # Number of unseen-combination validation samples per concept
)

# Ratio of compound scenarios (mixing 2 concepts, question-dependent).
# If this is 0, the model can answer correctly based purely on the context without reading the question.
# To prevent this, a significant portion of train/val consists of compound scenarios by default.
DEFAULT_COMPOUND_RATIO = (
    0.5  # Proportion of compound scenarios in the entire train/val dataset
)


def generate_dataset(
    seed: int = 42,
    train_per_concept: int = DEFAULT_TRAIN_SAMPLES_PER_CONCEPT,
    val_entity_per_concept: int = DEFAULT_VAL_ENTITY_SAMPLES_PER_CONCEPT,
    val_combo_per_concept: int = DEFAULT_VAL_COMBO_SAMPLES_PER_CONCEPT,
    compound_ratio: float = DEFAULT_COMPOUND_RATIO,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Generates a physics QA dataset consisting of $12 \text{ concepts} \times 2 \text{ polarities}$.

    In single-concept scenarios, there is only one correct answer per scenario, meaning the model
    could guess correctly even if it ignores the question text. To prevent this, we mix "compound scenarios"
    (where two concepts are mixed, and the correct answer changes depending on the entity asked about in the same scenario)
    at a rate specified by `compound_ratio`. This forces the model to actually read the question to answer correctly.

    For each concept:
      1. Split entities into train and unseen vocabularies.
      2. Create the full universe of combinations (train-vocab universe) via:
         $\text{entities}_{\text{train}} \times \text{templates} \times \text{questions}$
      3. Randomly sample `train_per_concept` examples from the universe for the training set.
      4. Sample `val_combo_per_concept` examples from the remaining train-vocab universe for unseen-combination validation.
      5. Sample `val_entity_per_concept` examples from the unseen entity universe for unseen-vocabulary validation.
      6. Additionally, generate compound scenarios mixing 2 concepts based on `compound_ratio`.

    Returns
    -------
    (train_records, val_records)
        Each element: {"scenario": str, "question": str, "label": int}
    """
    rng = random.Random(seed)

    train_records: list[dict] = []
    val_records: list[dict] = []

    total_universe = 0
    concept_entity_pools: dict = (
        {}
    )  # Record entity pools for generating compound scenarios

    for concept_name, spec in CONCEPTS.items():
        concept_entity_pools[concept_name] = {}
        for polarity in ("pos", "neg"):
            label_str = spec[f"{polarity}_label"]
            label_id = LABEL_TO_ID[label_str]
            entities = spec[f"{polarity}_entities"]
            templates = spec[f"{polarity}_templates"]
            questions = spec["questions"]

            train_entities, val_entities = _split_entities(entities, rng)
            concept_entity_pools[concept_name][
                polarity
            ] = train_entities  # Compound scenarios use only the training vocabulary

            # --- Full combinations of train-vocab (large universe)
            train_vocab_universe = _expand_combos(train_entities, templates, questions)
            total_universe += len(train_vocab_universe)

            rng.shuffle(train_vocab_universe)

            n_train = min(train_per_concept // 2, len(train_vocab_universe))
            sampled_train = train_vocab_universe[:n_train]
            remaining = train_vocab_universe[n_train:]

            for scenario, question, _ in sampled_train:
                train_records.append(
                    {
                        "scenario": scenario,
                        "question": question,
                        "label": label_id,
                        "_kind": "train",
                    }
                )

            # Unseen combination val (same words, unused sentence/question pairs)
            n_combo_val = min(val_combo_per_concept // 2, len(remaining))
            rng.shuffle(remaining)
            for scenario, question, _ in remaining[:n_combo_val]:
                val_records.append(
                    {
                        "scenario": scenario,
                        "question": question,
                        "label": label_id,
                        "_kind": "unseen_combo",
                    }
                )

            # Unseen entity val (entities never shown during training)
            val_vocab_universe = _expand_combos(val_entities, templates, questions)
            total_universe += len(val_vocab_universe)
            rng.shuffle(val_vocab_universe)
            n_entity_val = min(val_entity_per_concept // 2, len(val_vocab_universe))
            for scenario, question, _ in val_vocab_universe[:n_entity_val]:
                val_records.append(
                    {
                        "scenario": scenario,
                        "question": question,
                        "label": label_id,
                        "_kind": "unseen_entity",
                    }
                )

    # Compound scenarios (forcing dependency on the question):
    # Determine the number of compound scenarios based on the number of single-concept samples
    # in train/val to match the specified `compound_ratio`.
    if compound_ratio > 0:
        n_single_train = len(train_records)
        n_compound_train = int(n_single_train * compound_ratio / (1 - compound_ratio))
        compound_train = _generate_compound_examples(
            n_compound_train, rng, concept_entity_pools
        )
        for r in compound_train:
            r["_kind"] = "compound_train"
        train_records.extend(compound_train)

        n_single_val = len(val_records)
        n_compound_val = int(n_single_val * compound_ratio / (1 - compound_ratio))
        compound_val = _generate_compound_examples(
            n_compound_val, rng, concept_entity_pools
        )
        for r in compound_val:
            r["_kind"] = "compound_val"
        val_records.extend(compound_val)

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    if verbose:
        n_unseen_combo = sum(1 for r in val_records if r.get("_kind") == "unseen_combo")
        n_unseen_entity = sum(
            1 for r in val_records if r.get("_kind") == "unseen_entity"
        )
        n_compound_val = sum(1 for r in val_records if r.get("_kind") == "compound_val")
        n_compound_train = sum(
            1 for r in train_records if r.get("_kind") == "compound_train"
        )
        print(
            f"[PhysicsQA] Number of concepts: {len(CONCEPTS)}  Number of labels: {NUM_PHYSICS_LABELS}"
        )
        print(f"[PhysicsQA] Total combination space (universe): {total_universe:,}")
        print(
            f"[PhysicsQA] train: {len(train_records):,} records  "
            f"(single-concept={len(train_records)-n_compound_train:,} / compound={n_compound_train:,})"
        )
        print(
            f"[PhysicsQA] val: {len(val_records):,} records  "
            f"(unseen_combo={n_unseen_combo:,} / unseen_entity={n_unseen_entity:,} / compound={n_compound_val:,})"
        )

    train_records_clean = [
        {"scenario": r["scenario"], "question": r["question"], "label": r["label"]}
        for r in train_records
    ]
    val_records_clean = [
        {"scenario": r["scenario"], "question": r["question"], "label": r["label"]}
        for r in val_records
    ]

    return train_records_clean, val_records_clean


def generate_dataset_with_breakdown(seed: int = 42, **kwargs):
    """
    For analysis: Returns val split into unseen_entity, unseen_combo, and compound.
    Used when evaluating each category individually after fine-tuning.

    Returns
    -------
    (train_records, val_unseen_combo, val_unseen_entity, val_compound)
    """
    rng = random.Random(seed)
    train_per_concept = kwargs.get(
        "train_per_concept", DEFAULT_TRAIN_SAMPLES_PER_CONCEPT
    )
    val_entity_per_concept = kwargs.get(
        "val_entity_per_concept", DEFAULT_VAL_ENTITY_SAMPLES_PER_CONCEPT
    )
    val_combo_per_concept = kwargs.get(
        "val_combo_per_concept", DEFAULT_VAL_COMBO_SAMPLES_PER_CONCEPT
    )
    compound_ratio = kwargs.get("compound_ratio", DEFAULT_COMPOUND_RATIO)

    train_records: list[dict] = []
    val_unseen_combo: list[dict] = []
    val_unseen_entity: list[dict] = []
    concept_entity_pools: dict = {}

    for concept_name, spec in CONCEPTS.items():
        concept_entity_pools[concept_name] = {}
        for polarity in ("pos", "neg"):
            label_id = LABEL_TO_ID[spec[f"{polarity}_label"]]
            entities = spec[f"{polarity}_entities"]
            templates = spec[f"{polarity}_templates"]
            questions = spec["questions"]

            train_entities, val_entities = _split_entities(entities, rng)
            concept_entity_pools[concept_name][polarity] = train_entities
            train_vocab_universe = _expand_combos(train_entities, templates, questions)
            rng.shuffle(train_vocab_universe)

            n_train = min(train_per_concept // 2, len(train_vocab_universe))
            for scenario, question, _ in train_vocab_universe[:n_train]:
                train_records.append(
                    {"scenario": scenario, "question": question, "label": label_id}
                )

            remaining = train_vocab_universe[n_train:]
            rng.shuffle(remaining)
            n_combo_val = min(val_combo_per_concept // 2, len(remaining))
            for scenario, question, _ in remaining[:n_combo_val]:
                val_unseen_combo.append(
                    {"scenario": scenario, "question": question, "label": label_id}
                )

            val_vocab_universe = _expand_combos(val_entities, templates, questions)
            rng.shuffle(val_vocab_universe)
            n_entity_val = min(val_entity_per_concept // 2, len(val_vocab_universe))
            for scenario, question, _ in val_vocab_universe[:n_entity_val]:
                val_unseen_entity.append(
                    {"scenario": scenario, "question": question, "label": label_id}
                )

    val_compound: list[dict] = []
    if compound_ratio > 0:
        n_single_val = len(val_unseen_combo) + len(val_unseen_entity)
        n_compound_val = int(n_single_val * compound_ratio / (1 - compound_ratio))
        val_compound = _generate_compound_examples(
            n_compound_val, rng, concept_entity_pools
        )

    rng.shuffle(train_records)
    rng.shuffle(val_unseen_combo)
    rng.shuffle(val_unseen_entity)
    rng.shuffle(val_compound)

    return train_records, val_unseen_combo, val_unseen_entity, val_compound


# ============================================================
# Encoding (Reusing the same format as SNLI)
#    <s> scenario $ question </s>  ->  24-class classification
# ============================================================


class PhysicsQADataset:
    """
    Same interface as SNLIDataset (text_processing/snli_dataset.py).
    Places the scenario as the premise and the question as the hypothesis,
    reusing `encode_with_special()` directly.
    """

    def __init__(self, enc, max_seq_len: int = 64):
        self.encoder = enc
        self.max_seq_len = max_seq_len

    def prepare(self, records: list[dict], split_name: str = "train") -> dict:
        pad_id = self.encoder.encoder.get("<pad>", 0)
        all_ids, all_labels, all_lengths = [], [], []
        truncated = 0

        for rec in records:
            ids = self.encoder.encode_with_special(rec["scenario"], rec["question"])
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
                truncated += 1
            length = len(ids)
            ids = ids + [pad_id] * (self.max_seq_len - length)
            all_ids.append(ids)
            all_labels.append(rec["label"])
            all_lengths.append(length)

        print(
            f"[PhysicsQADataset] {split_name}: {len(records)} records  "
            f"truncated={truncated}  max_seq_len={self.max_seq_len}"
        )
        return {"input_ids": all_ids, "labels": all_labels, "lengths": all_lengths}
