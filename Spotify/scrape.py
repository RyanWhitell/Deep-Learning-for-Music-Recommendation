import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import wikipedia
import musicbrainzngs
musicbrainzngs.set_useragent('haamr', 1.0)

import requests
from bs4 import BeautifulSoup
import re

import h5py
import time
import datetime
from shutil import copyfile

import API_KEYS

SPOTIFY_CLIENT_ID = API_KEYS.SPOTIFY_CLIENT_ID
SPOTIFY_CLIENT_SECRET = API_KEYS.SPOTIFY_CLIENT_SECRET

LAST_FM_API_KEY = API_KEYS.LAST_FM_API_KEY
LAST_FM_SHARED_SECRET = API_KEYS.LAST_FM_SHARED_SECRET
LAST_FM_REGISTERED_TO = API_KEYS.LAST_FM_REGISTERED_TO
LAST_FM_API_KEY2 = API_KEYS.LAST_FM_API_KEY2
LAST_FM_SHARED_SECRET2 = API_KEYS.LAST_FM_SHARED_SECRET2
LAST_FM_REGISTERED_TO2 = API_KEYS.LAST_FM_REGISTERED_TO2

GENIUS_CLIENT_ID = API_KEYS.GENIUS_CLIENT_ID
GENIUS_CLIENT_SECRET = API_KEYS.GENIUS_CLIENT_SECRET
GENIUS_CLIENT_ACCESS_TOKEN = API_KEYS.GENIUS_CLIENT_ACCESS_TOKEN

MM_API_KEY = API_KEYS.MM_API_KEY

MB_CLIENT_ID = API_KEYS.MB_CLIENT_ID
MB_SECRET = API_KEYS.MB_SECRET

MAPS_API_KEY = API_KEYS.MAPS_API_KEY

import argparse

parser = argparse.ArgumentParser(description="scrapes various apis for music content")
parser.add_argument('-n', '--artists', default=0, help='number of artists from seed_artists.csv to scrape')
parser.add_argument('-i', '--seeds', default=None, help='injects seed artists via comma separated list')
args = parser.parse_args()


def inject_seed_artists(list_ids):
    f = h5py.File('./data.hdf5', 'a')

    if 'seed_artists' in f:
        seed_artists = f['seed_artists']
    else:
        seed_artists = f.create_group("seed_artists")

    added = []
    for i in list_ids:
        if i in seed_artists.keys():
            print(f'id {i} already in the list')
        else:
            seed_artists[i] = False
            added.append(i)

    f.close()
    return added

def get_next_artist():
    

if __name__=='__main__':
    print('File Start...')
    file_start = time.perf_counter()

    now = datetime.datetime.now()

    print('Backing up data file...')
    copyfile('./data.hdf5', './backups/' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_' + str(now.second) + '_' + 'data.hdf5')

    if args.seeds is not None:
        print(f'Adding {args.seeds} to seed_artists list...')
        try:
            added = inject_seed_artists(args.seeds.split(','))
            print(f'{added} added to seed_artists list')
        except Exception as e:
            print(f'Could not add seeds {args.seeds}.hdf5\n{e}')

    num_artists = args.artists
    while num_artists > 0:
        print('Getting next artist from seed_artists')
        next_artist_id = get_next_artist()
        print(f'Scraping info for artist with ID: {next_artist_id}')
    
