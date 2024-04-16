import os
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from typing import Optional
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
from tqdm import tqdm
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Main script
itunes_file = "/Users/renier.botha/Music/DJ/Library.xml"

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# Read iTunes library XML and convert to dataframe
def xml_to_dataframe(xml_file: str) -> pd.DataFrame:
    logger.info(f"Reading XML file {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_dicts = []
    for dict_elem in root.findall(".//dict/dict"):
        song_data = {}
        elements = list(dict_elem)
        for i in range(0, len(elements), 2):
            key = elements[i]
            value = elements[i + 1]
            if key.text in [
                "Name",
                "Artist",
                "Album",
                "Genre",
                "Comments",
                "Grouping",
                "Location",
            ]:
                song_data[key.text] = value.text
        song_data["id"] = str(uuid.uuid4())
        all_dicts.append(song_data)

    logger.info(f"Converted XML file to dataframe with {len(all_dicts)} rows")
    return pd.DataFrame(all_dicts)


import re

def clean_track_name(track_name: str) -> str:
    # Remove text between brackets
    cleaned_name = re.sub(r'\(.*?\)', '', track_name)

    # Remove special characters
    cleaned_name = re.sub(r'\W+', ' ', cleaned_name)
    
    return cleaned_name.strip()

def match_itunes_to_spotify(df: pd.DataFrame) -> pd.DataFrame:
    audio_features_list = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        query = f"track:{row['Name']} artist:{row['Artist']}"
        logger.info(f"Searching for track: {query}")
        results = sp.search(q=query, type="track", limit=1)
        tracks = results["tracks"]["items"]
        if tracks:
            track_id = tracks[0]["id"]
            audio_features = sp.audio_features([track_id])[0]
            audio_features["id"] = row["id"]
            audio_features["matched_name"] = tracks[0]["name"]
            audio_features_list.append(audio_features)
        else:
            logger.warning(f"No tracks found for query: {query}")
            # Retry with cleaned track name
            cleaned_name = clean_track_name(row['Name'])
            query = f"track:{cleaned_name} artist:{row['Artist']}"
            logger.info(f"Retrying with cleaned track name: {query}")
            results = sp.search(q=query, type="track", limit=1)
            tracks = results["tracks"]["items"]
            if tracks:
                track_id = tracks[0]["id"]
                audio_features = sp.audio_features([track_id])[0]
                audio_features["id"] = row["id"]
                audio_features["matched_name"] = tracks[0]["name"]
                audio_features_list.append(audio_features)
            else:
                audio_features_list.append({"id": row["id"]})

    df_audio_features = pd.DataFrame(audio_features_list)
    final_df = pd.merge(df, df_audio_features, how="left", on="id")
    return final_df


if __name__ == "__main__":
    df = xml_to_dataframe(itunes_file).sample(100)
    final_df = match_itunes_to_spotify(df)

    logger.info("Script completed successfully")

    final_df.to_csv("itunes_spotify_matched.csv")
