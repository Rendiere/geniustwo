# Setup

```
pyenv shell 3.10
pyenv local

poetry install
poetry shell

```

# Current Issues

- Search is missing about 40% of tracks
- Need some way of re-trying the mis-matched songs
    - For example "track:Visions (feat. Kirsty Hawkshaw) artist: Ian Pooley" does not match, but just "track:Visions artist: Ian Pooley" _does_match.
    - In this case, it's a good match, but in some cases it might not be -- for example when you are interested in the dub or remix versions
- Should also double check matched song titles with what was searched
    - Add additional columns to DB: matched_title, matched_artist 