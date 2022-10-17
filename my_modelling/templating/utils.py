import json
import hashlib
import humanhash
import copy

class Hasher:
    """
    Calculates the sha256 hash and human readable hash of a string or dictionary.
    """
    def __init__(self, data):
        """Create hash from data."""
        self.hash = Hasher.calc_hash(data)

    @property
    def human_hash(self):
        """Calculates a human readable hash of a sha256 hash."""
        return humanhash.humanize(self.hash, words = 3)

    @staticmethod
    def calc_hash(data):
        """Calculates the sha256 hash of a string or dictionary."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)

        try:
            hash = hashlib.sha256(data.encode()).hexdigest()
        except:
            raise ValueError("Data must be a string or dictionary.")

        return hash

    def __str__(self):
        """Return the hash."""
        return self.hash

def dict_has_nones(data):
    """Loops through dictionary and returns boolean if any of the values are None."""
    for _, value in data.items():
        if isinstance(value, dict):
            if dict_has_nones(value):
                return True

        if value is None:
            return True

    return False

def remove_ids(data):
    """Removes all keys that are an ID for a dictionary."""
    data = copy.deepcopy(data)
    id_vars = ["ID", "ID_human"]
    for key in id_vars:
        if key in data:
            data.pop(key)
    
    # Remove from all nested dictionaries
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = remove_ids(value)

    return data