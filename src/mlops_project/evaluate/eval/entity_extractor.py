import spacy

nlp = spacy.load("en_core_web_sm")

# Simple domain keyword lists
WAYPOINT_KEYWORDS = [
    "zone",
    "sector",
    "ridge",
    "hill",
    "checkpoint",
    "area",
    "division",
    "line",
]
RESOURCE_KEYWORDS = [
    "engine",
    "truck",
    "drone",
    "crew",
    "helicopter",
    "strike",
    "equipment",
]
ACTION_VERBS = [
    "evacuate",
    "deploy",
    "extinguish",
    "suppress",
    "monitor",
    "establish",
    "move",
]


# Simple phrase cleanup
def clean_phrase(phrase: str) -> str | None:
    phrase = phrase.strip()

    if len(phrase.split()) > 4:  # too long
        return None
    if phrase.isdigit():  # just a number
        return None
    if len(set(phrase.lower().split())) <= 1:  # repeated word
        return None
    if len(phrase) < 3:  # too short
        return None

    return phrase


def extract_entities(text):
    doc = nlp(text)

    waypoints = []
    resources = []
    actions = []

    # Use NER for initial extraction
    for ent in doc.ents:
        token = ent.text.lower()
        cleaned = clean_phrase(ent.text)
        if not cleaned:
            continue

        if any(k in token for k in WAYPOINT_KEYWORDS):
            waypoints.append(cleaned)
        elif any(k in token for k in RESOURCE_KEYWORDS):
            resources.append(cleaned)

    # Use POS + verb matching for actions
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in ACTION_VERBS:
            cleaned = clean_phrase(token.lemma_)
            if cleaned:
                actions.append(cleaned)

    return {
        "waypoints": list(set(waypoints)),
        "actions": list(set(actions)),
        "resources": list(set(resources)),
    }
