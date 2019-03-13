import pandas as pd
import pickle

MSD_ARTIST_LOCATION = pd.read_csv(
    './artist_location.txt', 
    sep='<SEP>', 
    header=None, 
    names=['MSD_Artists_ID', 'lat', 'lng', 'name', 'place'],
    engine='python'
)
for index, row in MSD_ARTIST_LOCATION.iterrows():
    MSD_ARTIST_LOCATION.loc[index] = [row['MSD_Artists_ID'], row['lat'], row['lng'], row['name'].lower(), str(row['place']).lower()]

WORLD_CITIES = pd.read_csv(
    './worldcities.csv'
)
int_lat, int_lng, mesh = [], [], []
for index, row in WORLD_CITIES.iterrows():
    int_lat.append(int(row['lat']))
    int_lng.append(int(row['lng']))
    mesh.append((str(row['city']) + str(row['admin_name']) + str(row['country'])).replace(' ', ''))
    
WORLD_CITIES['int_lat'] = int_lat
WORLD_CITIES['int_lng'] = int_lng
WORLD_CITIES['mesh'] = mesh

LYRICS1 = pd.read_csv(
    './lyrics.csv'
)
LYRICS2 = pd.read_csv(
    './lyrics1.csv'
)
LYRICS3 = pd.read_csv(
    './lyrics2.csv'
)
LYRICS4 = pd.read_csv(
    './songdata.csv'
)

artist, song, lyrics = [], [], []
for _, row in LYRICS1.iterrows():
    artist.append(str(row['artist']).replace('-', ' '))
    song.append(str(row['song']).replace('-', ' '))
    lyrics.append(row['lyrics'])
    
for _, row in LYRICS2.iterrows():
    artist.append(str(row['Band']).lower())
    song.append(str(row['Song']).lower())
    lyrics.append(row['Lyrics'])
    
for _, row in LYRICS3.iterrows():
    artist.append(str(row['Band']).lower())
    song.append(str(row['Song']).lower())
    lyrics.append(row['Lyrics'])
     
for _, row in LYRICS4.iterrows():
    artist.append(str(row['artist']).lower())
    song.append(str(row['song']).lower())
    lyrics.append(row['text'])

LYRICS = pd.DataFrame()
LYRICS['artist'] = artist
LYRICS['song'] = song
LYRICS['lyrics'] = lyrics

LYRICS = LYRICS.loc[set(LYRICS.index.values) - set(LYRICS.loc[LYRICS.lyrics.isnull()].index.values)]
LYRICS = LYRICS.drop_duplicates(['artist', 'song'])

with open('./external_data.pickle', 'wb') as f:
    save = {
        'LYRICS': LYRICS,
        'WORLD_CITIES': WORLD_CITIES,
        'MSD_ARTIST_LOCATION': MSD_ARTIST_LOCATION
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    del save