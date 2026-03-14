# =================================================================
# matching.py
# W210 Capstone — Music-to-Ad Matching Engine
# =================================================================

import json
import re
import numpy as np
import pandas as pd
import joblib
import boto3
import anthropic

# =================================================================
# CONSTANTS
# =================================================================

BUCKET = 'mids-capstone-music-ad-matching-2026'
POPULARITY_PENALTY = 0.20

TEMPO_MIN = 0.0
TEMPO_MAX = 250.0

TEMPO_MAP = {'Slow': 0.25, 'Medium': 0.50, 'Fast': 0.75}
ENERGY_MAP = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
MOOD_MAP = {'Positive': 0.8, 'Neutral': 0.5, 'Serious': 0.2}

GENRE_INDUSTRY_MAP = {
    'Electronic':          ['Tech', 'Entertainment'],
    'Chiptune':            ['Tech', 'Entertainment'],
    'Sound Art':           ['Tech', 'Entertainment'],
    'Rock':                ['Automotive', 'Entertainment'],
    'Punk':                ['Entertainment', 'Retail'],
    'Post-Punk':           ['Entertainment', 'Retail'],
    'Post-Rock':           ['Automotive', 'Entertainment'],
    'Metal':               ['Automotive', 'Entertainment'],
    'Psych-Rock':          ['Entertainment', 'Retail'],
    'Indie-Rock':          ['Retail', 'Entertainment', 'F&B'],
    'Pop':                 ['Retail', 'F&B', 'Healthcare', 'Entertainment'],
    'Hip-Hop':             ['Retail', 'Entertainment', 'Automotive'],
    'Trip-Hop':            ['Retail', 'Finance', 'Healthcare'],
    'Folk':                ['F&B', 'Healthcare', 'Retail'],
    'Psych-Folk':          ['F&B', 'Entertainment', 'Retail'],
    'Old-Time / Historic': ['F&B', 'Finance', 'Retail'],
    'Jazz':                ['Finance', 'F&B', 'Healthcare'],
    'Blues':               ['F&B', 'Entertainment'],
    'Classical':           ['Finance', 'Healthcare', 'F&B'],
    'Soundtrack':          ['Automotive', 'Entertainment', 'Finance'],
    'International':       ['F&B', 'Entertainment', 'Retail'],
    'Kid-Friendly':        ['F&B', 'Healthcare', 'Retail'],
    'Compilation':         ['Retail', 'Entertainment']
}

# =================================================================
# LOAD MODEL ARTIFACTS FROM S3
# =================================================================

def load_artifacts(api_key: str):
    """Load all model artifacts from S3. Call once at app startup."""
    s3 = boto3.client('s3')

    # Download and load scaler
    s3.download_file(BUCKET, 'models/scaler_v3.pkl', '/tmp/scaler_v3.pkl')
    scaler = joblib.load('/tmp/scaler_v3.pkl')

    # Download and load PCA
    s3.download_file(BUCKET, 'models/pca_v3.pkl', '/tmp/pca_v3.pkl')
    pca = joblib.load('/tmp/pca_v3.pkl')

    # Download and load V4 weights
    s3.download_file(BUCKET, 'models/v4_weights.json', '/tmp/v4_weights.json')
    with open('/tmp/v4_weights.json') as f:
        weights = json.load(f)

    # Download and load track data
    s3.download_file(BUCKET, 'processed-data/fma/fma_pre_z_filtered_avail.csv', '/tmp/fma_avail.csv')
    df_tracks = pd.read_csv('/tmp/fma_avail.csv', low_memory=False)

    # Normalize tempo
    df_tracks['echonest_audio_features_tempo_norm'] = (
        (df_tracks['echonest_audio_features_tempo'] - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    )

    # Build librosa PCA matrix
    librosa_cols = [col for col in df_tracks.columns if col.startswith(('mfcc', 'chroma', 'spectral', 'tonnetz', 'zcr'))]
    df_librosa_scaled = scaler.transform(df_tracks[librosa_cols])
    df_pca = pd.DataFrame(
        pca.transform(df_librosa_scaled),
        index=df_tracks.index,
        columns=[f'pca_{i+1:03d}' for i in range(pca.n_components_)]
    )

    # Init Anthropic client
    claude_client = anthropic.Anthropic(api_key=api_key)

    return {
        'df_tracks': df_tracks,
        'df_pca': df_pca,
        'weights': weights,
        'claude_client': claude_client
    }

# =================================================================
# LLM FEATURE DERIVATION
# =================================================================

def get_llm_features(description: str, claude_client) -> dict:
    """Derive danceability and acousticness from ad description via Claude API."""
    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""You are a music supervisor choosing background music for an advertisement.
Given this ad description, estimate these audio features for the IDEAL BACKGROUND MUSIC:

- danceability: how groovy/rhythmic should the music be (0=ambient/slow, 1=highly danceable)
- acousticness: should music be acoustic or electronic (0=fully electronic, 1=fully acoustic)

Ad description: {description}

Think carefully about the specific tone, energy, and context of this ad.
Return ONLY a JSON object with float values between 0.0 and 1.0, no other text."""
        }]
    )
    text = message.content[0].text.strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {'danceability': 0.5, 'acousticness': 0.5}  # fallback

# =================================================================
# MATCHING
# =================================================================

def get_valid_genres(industry: str) -> list:
    return [genre for genre, industries in GENRE_INDUSTRY_MAP.items()
            if industry in industries]

def match_tracks(
    energy: int,
    tempo: str,
    mood: str,
    industry: str,
    ad_description: str,
    artifacts: dict,
    lyrics_preference: str = 'No Preference',  # 'Lyrics', 'No Lyrics', 'No Preference'
    genre_override: list = None,
    top_n: int = 3
) -> list:
    """
    Run V4 matching for a new ad input.

    Parameters:
        energy: 1-5
        tempo: 'Slow', 'Medium', 'Fast'
        mood: 'Positive', 'Neutral', 'Serious'
        industry: one of 8 industries
        ad_description: free text ad description
        artifacts: loaded model artifacts from load_artifacts()
        lyrics_preference: 'Lyrics', 'No Lyrics', 'No Preference'
        genre_override: optional list of genre strings to override industry filter
        top_n: number of tracks to return

    Returns:
        list of dicts with top_n track recommendations
    """
    df_tracks = artifacts['df_tracks'].copy()
    df_pca = artifacts['df_pca']
    weights = artifacts['weights']
    claude_client = artifacts['claude_client']

    # Map inputs to numeric targets
    target_energy = ENERGY_MAP[energy]
    target_tempo = TEMPO_MAP[tempo]
    target_valence = MOOD_MAP[mood]

    # Derive LLM features from description
    llm_features = get_llm_features(ad_description, claude_client)
    llm_danceability = llm_features.get('danceability', 0.5)
    llm_acousticness = llm_features.get('acousticness', 0.5)

    # Apply genre filter
    if genre_override:
        valid_genres = genre_override
    else:
        valid_genres = get_valid_genres(industry)

    if valid_genres and industry != 'Other':
        df_filtered = df_tracks[df_tracks['track_genres_name'].isin(valid_genres)].copy()
    else:
        df_filtered = df_tracks.copy()

    if len(df_filtered) < 10:
        df_filtered = df_tracks.copy()

    # Apply lyrics filter
    if lyrics_preference == 'No Lyrics':
        df_filtered = df_filtered[
            df_filtered['echonest_audio_features_instrumentalness'] > 0.8
        ].copy()
        if len(df_filtered) < 10:
            df_filtered = df_tracks.copy()  # fallback if too few tracks

    df_pca_filtered = df_pca.loc[df_filtered.index]

    # Compute weighted distance (V4 weights)
    scores = np.zeros(len(df_filtered))
    scores += weights['energy'] * (df_filtered['echonest_audio_features_energy'].values - target_energy) ** 2
    scores += weights['tempo'] * (df_filtered['echonest_audio_features_tempo_norm'].values - target_tempo) ** 2
    scores += weights['valence'] * (df_filtered['echonest_audio_features_valence'].values - target_valence) ** 2
    scores += weights['danceability'] * (df_filtered['echonest_audio_features_danceability'].values - llm_danceability) ** 2
    scores += weights['acousticness'] * (df_filtered['echonest_audio_features_acousticness'].values - llm_acousticness) ** 2

    pca_centroid = df_pca_filtered.mean().values
    pca_distances = np.sum((df_pca_filtered.values - pca_centroid) ** 2, axis=1)
    pca_distances = pca_distances / pca_distances.max()
    scores += weights['pca'] * pca_distances

    scores = np.sqrt(scores)

    # Popularity penalty
    high_mask = df_filtered['popularity_bucket'].values == 'High'
    scores[high_mask] *= (1 + POPULARITY_PENALTY)

    df_filtered = df_filtered.copy()
    df_filtered['match_score'] = scores
    top_matches = df_filtered.nsmallest(top_n, 'match_score')

    # Return clean result list
    results = []
    for _, row in top_matches.iterrows():
        results.append({
            'artist': row['artist_name'],
            'title': row['track_title'],
            'genre': row['track_genres_name'],
            'fma_url': row['fma_album_url'],
            'match_score': round(row['match_score'], 4),
            'popularity': row['popularity_bucket']
        })

    return results