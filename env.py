from os import environ as env
from pathlib import Path

PROJECT_NAME = "" # Name of the project used to name the directory in the cache
HOME = "" # Path to the project

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

DATA_PATH = "" # Path to the datasets

env['HYDRA_FULL_ERROR'] = '1'

# Your OpenAI API key
OPENAI_API_KEY = ""

# Your Mistral AI API key
MISTRALAIKEY = ""

