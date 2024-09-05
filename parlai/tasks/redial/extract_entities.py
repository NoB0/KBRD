import json
import pickle as pkl
import re
from collections import defaultdict
from typing import List, Tuple

import requests
from tqdm import tqdm

# Set up a local dbpedia-spotlight docker https://github.com/dbpedia-spotlight/spotlight-docker
# DBPEDIA_SPOTLIGHT_ADDR = " http://0.0.0.0:2222/rest/annotate"
DBPEDIA_SPOTLIGHT_ADDR = "https://api.dbpedia-spotlight.org/en/annotate"
SPOTLIGHT_CONFIDENCE = 0.5
SPOTLIGHT_SUPPORT = 50
MOVIE_TYPES = [
    "DBpedia:Film",
    "DBpedia:Movie",
]


def _id2dbpedia(movie_id):
    pass


def _text2entities(text: str) -> List[Tuple[str, bool]]:
    """Extract entities from text using DBPedia Spotlight.

    Args:
        text: Text to analyze.

    Returns:
        List of tuples with entity URI and whether it is a movie or not.
    """
    headers = {"accept": "application/json"}
    params = {
        "text": text,
        "confidence": SPOTLIGHT_CONFIDENCE,
        "support": SPOTLIGHT_SUPPORT,
    }

    response = requests.get(
        DBPEDIA_SPOTLIGHT_ADDR, headers=headers, params=params
    )
    response = response.json()
    resources = list()
    if "Resources" in response:
        for x in response["Resources"]:
            uri = f"<{x['@URI']}>"
            if any([t in x["@types"] for t in MOVIE_TYPES]):
                resources.append((uri, True))
            else:
                resources.append((uri, False))
    return resources


def _tags(split, tags_dict, text_dict):
    path = f"data/redial/{split}_data.jsonl"
    instances = []
    with open(path) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))

    print(split, len(instances))

    num_tags = 0
    pattern = re.compile("@\d+")
    for instance in tqdm(instances):
        initiator_id = instance["initiatorWorkerId"]
        respondent_id = instance["respondentWorkerId"]
        messages = instance["messages"]
        for message in messages:
            if message["text"] != "":
                ent_tuples = _text2entities(message["text"])
                tags = [ent_tuple[0] for ent_tuple in ent_tuples]
                text_dict[message["text"]] = tags
                num_tags += len(tags)
                for tag in tags:
                    tags_dict[tag] += 1


if __name__ == "__main__":
    tags_dict = defaultdict(int)
    text_dict = {}
    for split in ["train", "valid", "test"]:
        _tags(split, tags_dict, text_dict)

    print(len(tags_dict))

    pkl.dump(text_dict, open("data/redial/text_dict_confidence_0.1.pkl", "wb"))
    pkl.dump(
        tags_dict, open("data/redial/entity_dict_confidence_0.1.pkl", "wb")
    )
