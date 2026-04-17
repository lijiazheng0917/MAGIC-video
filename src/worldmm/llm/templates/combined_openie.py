"""Combined NER + Triple extraction in a single LLM call."""

combined_system = """Your task is to extract named entities and construct an RDF (Resource Description Framework) graph from the given paragraph.

Respond with a JSON object containing two keys:
1. "named_entities": A list of named entities found in the paragraph.
2. "triples": A list of RDF triples, each as a 3-element list [subject, predicate, object].

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities.
- When resolving pronouns, if the pronoun refers to the first-person (e.g., I, me, my), keep it as "I" instead of replacing with terms like "speaker" or "narrator". For other pronouns, clearly resolve them to their specific names to maintain clarity.
"""

one_shot_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_output = """{
    "named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"],
    "triples": [
        ["Radio City", "located in", "India"],
        ["Radio City", "is", "private FM radio station"],
        ["Radio City", "started on", "3 July 2001"],
        ["Radio City", "plays songs in", "Hindi"],
        ["Radio City", "plays songs in", "English"],
        ["Radio City", "forayed into", "New Media"],
        ["Radio City", "launched", "PlanetRadiocity.com"],
        ["PlanetRadiocity.com", "launched in", "May 2008"],
        ["PlanetRadiocity.com", "is", "music portal"],
        ["PlanetRadiocity.com", "offers", "news"],
        ["PlanetRadiocity.com", "offers", "videos"],
        ["PlanetRadiocity.com", "offers", "songs"]
    ]
}"""

prompt_template = [
    {"role": "system", "content": combined_system},
    {"role": "user", "content": one_shot_paragraph},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": "${passage}"}
]
