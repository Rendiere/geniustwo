import os
import pandas as pd
import xml.etree.ElementTree as ET
import logging
import uuid
import urllib.parse
import librosa

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def format_file_path(file_path: str) -> str:
    # Remove 'file:///' prefix
    if file_path.startswith("file:///"):
        file_path = file_path[7:]
    # Decode URL-encoded characters
    file_path = urllib.parse.unquote(file_path)
    return file_path

def xml_to_dataframe(xml_file: str) -> pd.DataFrame:
    logger.info(f'Reading XML file {xml_file}')
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_dicts = []
    for dict_elem in root.findall(".//dict/dict"):
        song_data = {}
        elements = list(dict_elem)
        for i in range(0, len(elements), 2):
            key = elements[i]
            value = elements[i + 1]
            if key.text in ["Name", "Artist", "Album", "Genre", "Comments", "Grouping", "Location"]:
                song_data[key.text] = value.text
        song_data["id"] = str(uuid.uuid4())
        all_dicts.append(song_data)

    logger.info(f'Converted XML file to dataframe with {len(all_dicts)} rows')
    return pd.DataFrame(all_dicts).dropna(subset=['Location'])

def load_audio_file(file_path):
    try:
        file_path = format_file_path(file_path)
        if not os.path.isfile(file_path):
            print(f"File does not exist: {file_path}")
            return None, None
        data, sr = librosa.load(file_path, sr=None)
        return data, sr
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None, None
