import json

_config = None

def load_config() -> json:
    global _config
    if _config is None:
        with open('config.json', 'r') as f:
            _config = json.load(f)
    return _config