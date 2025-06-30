from os import environ as env
from pathlib import Path

PROJECT_NAME = "linearly-interpretable-concept-embedding-model" # Name of the project used to name the directory in the cache
HOME = "/home/fdesantis/projects/Linearly-Interpretable-Concept-Embedding-Model-for-Text" # Path to the project

CACHE = Path(
    env.get(
        f"{PROJECT_NAME.upper()}_CACHE",
        Path(
            env.get("XDG_CACHE_HOME", Path("~", ".cache")),
            PROJECT_NAME,
        ),
    )
).expanduser()
CACHE.mkdir(exist_ok=True)

DATA_PATH = "/home/fdesantis/datasets" # Path to the datasets

env['HYDRA_FULL_ERROR'] = '1'

# Your OpenAI API key
OPENAI_API_KEY = "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"

# Your Mistral AI API key
MISTRALAIKEY = ""

