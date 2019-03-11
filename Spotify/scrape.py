import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import wikipedia
import musicbrainzngs

import urllib.request
import urllib.request as urllib2
import urllib.parse
import json
import requests
from bs4 import BeautifulSoup
import re

import h5py
import time
import datetime
import pandas as pd
import pickle
import pycountry

import random
from shutil import copyfile
import logging
import argparse
import os  
import glob

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

LYRICS_FOUND_BY_MM = '681F1AF6-8A1A-4493-8020-E44E2006ADB1***LYRICS_FOUND_BY_MM***361E1163-EE9C-444D-874D-7E0D438EF459'

NOW = datetime.datetime.now()
NOW = str(NOW.month) + '_' + str(NOW.day) + '_' + str(NOW.hour) + '_' + str(NOW.minute)
logging.basicConfig(filename='./dumps/' + NOW + '_.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

musicbrainzngs.set_useragent('haamr', 1.0)
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
SPOTIFY = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

parser = argparse.ArgumentParser(description="scrapes various apis for music content")
parser.add_argument('-n', '--num-seed-artists', default=0, help='number of seed_artists to scrape')
parser.add_argument('-s', '--seeds', default=None, help='injects seed artists via comma separated list')
parser.add_argument('-t', '--seeds-top', default=False, help='inject seeds at the top of the list')
parser.add_argument('-r', '--seeds-reset', default=False, help='reset seed artists that failed so they can run a second time')
parser.add_argument('-u', '--set-seed-unscraped', default=None, help='sets a seed as unscraped')
parser.add_argument('-p', '--print-top-genres', default=False, help='prints top genres sorted by count')
args = parser.parse_args()

####### Utility #######
def printlog(message, e=False):
    print(message)
    if e:
        logging.exception(message)
    else:
        logging.info(message)

def get_dataframes():
    if os.path.isfile('./data.pickle'):
        with open('./data.pickle', 'rb') as f:
            save = pickle.load(f)
            artists = save['artists']
            future_artists = save['future_artists']
            seed_artists = save['seed_artists']
            albums = save['albums']
            tracks = save['tracks']
            del save
    else:
        # id: {name: '', genres: [], related: [], lat: 0.0, lng: 0.0}
        col =  ['name', 'genres', 'related', 'lat', 'lng']
        artists  = pd.DataFrame(columns=col)
        future_artists = pd.DataFrame(columns=col)

        # id: {has_been_scraped: bool}
        col =  ['has_been_scraped']
        seed_artists = pd.DataFrame(columns=col)

        # id: {artist_id: '', name: '', release_date: '', release_date_precision: ''}
        col =  ['artist_id', 'name', 'release_date', 'release_date_precision']
        albums = pd.DataFrame(columns=col)

        # id: {artist_id: '', album_id: '', name: ''}
        col =  ['artist_id', 'album_id', 'name']
        tracks = pd.DataFrame(columns=col)

    return artists, future_artists, seed_artists, albums, tracks

def backup_dataframes():
    if os.path.isfile('./data.pickle'):
        printlog('Backing up data file...')
        copyfile('./data.pickle', './backups/' + NOW + '_' + 'data.pickle')

def save_dataframes(artists, future_artists, seed_artists, albums, tracks):
    backup_dataframes()

    with open('./data.pickle', 'wb') as f:
        save = {
            'artists': artists,
            'future_artists': future_artists,
            'seed_artists': seed_artists,
            'albums': albums,
            'tracks': tracks
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        del save

    printlog('data.pickle saved succesfully!')

def print_top_genres(df, top_genres):
    genre_counts = pd.DataFrame(columns=['count'])
    for genre_list in df.genres.values:
        for genre in genre_list:
            if genre in genre_counts.index:
                genre_counts.loc[genre] += 1
            else:
                genre_counts.loc[genre] = 1

    genre_counts = genre_counts.sort_values(by=['count'])
    
    top_genres_sorted = []
    for genre in genre_counts.index:
        if genre in top_genres:
            top_genres_sorted.append(genre)

    has_top_genre = []
    top_genre_counts = pd.DataFrame(columns=['count'])
    for genre in top_genres_sorted:
        top_genre_counts.loc[genre] = 0

    for genre in top_genres_sorted:
        for index, row in df.iterrows():
            if index not in has_top_genre:
                if genre in row['genres']:
                    has_top_genre.append(index)
                    top_genre_counts.loc[genre] += 1

    for genre in top_genres_sorted:
        for index, row in df.iterrows():
            if index not in has_top_genre:
                if genre in ' '.join(row['genres']):
                    has_top_genre.append(index)
                    top_genre_counts.loc[genre] += 1  
                    
    genre_counts = pd.DataFrame(columns=['count'])
    for genre_list in df.loc[set(df.index) - set(has_top_genre)].genres.values:
        for genre in genre_list:
            if genre in genre_counts.index:
                genre_counts.loc[genre] += 1
            else:
                genre_counts.loc[genre] = 1
                
                
    genre_counts = genre_counts.sort_values(by=['count'])
    printlog(genre_counts)

    top_genre_counts = top_genre_counts.sort_values(by=['count'])
    printlog(top_genre_counts)

def reset_failed_seed_artists(seed, artists):
    seed.loc[set(seed.loc[seed.has_been_scraped == True].index) - set(artists.index)] = False
   

####### Data #######
def inject_seed_artists(df, list_ids, top=False):
    if top:
        df_top = pd.DataFrame(columns=['has_been_scraped'])
        for i in list_ids:
            df_top.loc[i] = False
        return pd.concat([df_top, df])
    else:
        for i in list_ids:
            if i not in df.index:
                df.loc[i] = False
            
def get_next_seed_artist(seed_artists):
    try:
        return seed_artists.loc[seed_artists.has_been_scraped == False].index.values[0]
    except IndexError:
        return -1

def mark_seed_as_scraped(df, seed_artist_id):
    df.loc[seed_artist_id] = True

def mark_seed_as_unscraped(df, seed_artist_id):
    df.loc[seed_artist_id] = False

def add_artist(df, artist_id, name, genres, related, lat, lng):
    if artist_id not in df.index:
        df.loc[artist_id] = [name, genres, related, lat, lng]

def add_track(df, track_id, artist_id, album_id, name):
    if track_id not in df.index:
        df.loc[track_id] = [artist_id, album_id, name]

def add_albums(df, album_id, artist_id, name, release_date, release_date_precision):
    if album_id not in df.index:
        df.loc[album_id] = [artist_id, name, release_date, release_date_precision]

####### Artist #######
def fr_get_related(res):
    col =  ['name', 'genres', 'related', 'lat', 'lng']
    artists  = pd.DataFrame(columns=col)

    for artist in res['artists']:
        df = fr_get_artist_metadata(res=artist)
        artists.loc[artist['id']] = df.loc[artist['id']]

    return artists

def fr_get_artist_metadata(res):
    col =  ['name', 'genres', 'related', 'lat', 'lng']
    artist  = pd.DataFrame(columns=col)
    
    printlog(
        str(res['id']) +
        ' : ' + str(res['name']) +
        ' : ' + str(res['genres']) +
        ' : ??' +  ' : ??'
    )
    
    artist.loc[res['id']] = [res['name'], res['genres'], None, None, None]

    return artist

####### Lyrics #######
def get_lyrics_genius(song_title, artist_name):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + GENIUS_CLIENT_ACCESS_TOKEN}
    search_url = base_url + '/search'
    data = {'q': song_title + ' ' + artist_name}
    response = requests.get(search_url, data=data, headers=headers)

    json = response.json()
    remote_song_info = None
    hits = json['response']['hits']
    
    for hit in hits:
        if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
            remote_song_info = hit
            break
    
    if remote_song_info:
        song_url = remote_song_info['result']['url']
        page = requests.get(song_url)
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics = html.find('div', class_='lyrics').get_text()
        lyrics = re.sub(re.compile(r'\[Verse \d\]|\[Verse\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Chorus \d\]|\[Chorus\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Pre-Chorus \d\]|\[Pre-Chorus\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Bridge \d\]|\[Bridge\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Outro \d\]|\[Outro\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Pre-Chorus \d\]|\[Pre-Chorus\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Hook \d\]|\[Hook\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Sample \d\]|\[Sample\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Refrain \d\]|\[Refrain\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'\[Intro \d\]|\[Intro\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile(r'^(\n)*|(\n)*$'), '', lyrics)
        return lyrics

def get_lyrics_wikia(song_title, artist_name):
    url = 'http://lyric-api.herokuapp.com/api/find/' + artist_name.replace(' ', '%20') + '/' + song_title.replace(' ', '%20')
    return json.load(urllib2.urlopen(url))['lyric']

def get_lyrics_az(song_title, artist_name):
    artist_name = artist_name.lower()
    song_title = song_title.lower()
    
    # remove all except alphanumeric characters from artist_name and song_title
    artist_name = re.sub('[^A-Za-z0-9]+', "", artist_name)
    song_title = re.sub('[^A-Za-z0-9]+', "", song_title)
    
    # remove starting 'the' from artist_name e.g. the who -> who
    if artist_name.startswith("the"):    
        artist_name = artist_name[3:]
        
    url = "http://azlyrics.com/lyrics/"+artist_name+"/"+song_title+".html"
    
    content = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(content, 'html.parser')
    lyrics = str(soup)
    # lyrics lies between up_partition and down_partition
    up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
    down_partition = '<!-- MxM banner -->'
    lyrics = lyrics.split(up_partition)[1]
    lyrics = lyrics.split(down_partition)[0]
    lyrics = lyrics.replace('<br>','').replace('<br/>','').replace('</br>','').replace('</div>','').strip()
    
    return lyrics  

def get_lyrics_mm(song_title, artist_name):
    base_url = "https://api.musixmatch.com/ws/1.1/" + "matcher.lyrics.get" + "?format=json&callback=callback"
    artist_search = "&q_artist=" + artist_name
    song_search = "&q_track=" + song_title
    api = "&apikey=" + MM_API_KEY
    api_call = base_url + artist_search + song_search + api
    request = requests.get(api_call)
    data = request.json()
    data = data['message']['body']
    return data['lyrics']['lyrics_body'][:-70]
   
def get_track_lyrics(song_title, artist_name):
    printlog('Try Genius...')
    try:
        lyrics = get_lyrics_genius(song_title, artist_name)
        if len(lyrics) == 0:
            raise Exception('Lyrics empty')
    except Exception:
        printlog('Genius lyrics not found, try AZ...', e=True)
        try:
            lyrics = get_lyrics_az(song_title, artist_name)
            if len(lyrics) == 0:
                raise Exception('Lyrics empty')
        except Exception:
            printlog('AZ lyrics not found, try wikia...', e=True)
            try:
                lyrics = get_lyrics_wikia(song_title, artist_name)
                if len(lyrics) == 0:
                    raise Exception('Lyrics empty')
            except Exception:
                printlog('wikia lyrics not found, try mm...', e=True)
                try:
                    lyrics = get_lyrics_mm(song_title, artist_name)
                    if len(lyrics) == 0:
                        raise Exception('Lyrics empty')
                    lyrics = lyrics + '\n\n' + LYRICS_FOUND_BY_MM
                except Exception:
                    printlog('No lyrics found, exit', e=True)
                    return None

    return lyrics

####### Track #######
def fr_get_top_tracks_albums(country, artist_name, artist_id):  
    res_home = SPOTIFY.artist_top_tracks(seed_artist_id, country=country)
    res_US = SPOTIFY.artist_top_tracks(seed_artist_id, country='US')

    # id: {artist_id: '', name: '', release_date: '', release_date_precision: ''}
    col =  ['artist_id', 'name', 'release_date', 'release_date_precision']
    albums = pd.DataFrame(columns=col)

    # id: {artist_id: '', album_id: '', name: ''}
    col =  ['artist_id', 'album_id', 'name']
    tracks = pd.DataFrame(columns=col)

    lyrics_list = {}
    previews = {}
    album_covers = {}

    for track in res_home['tracks']:
        if track['preview_url'] is None:
            break
            
        try:
            lyrics = get_track_lyrics(track['name'], artist_name)
        except Exception:
            continue

        if lyrics is not None:
            printlog(
                str(track['id']) +
                ' : ' + str(track['name'])
            )
            tracks.loc[track['id']] = [artist_id, track['album']['id'], track['name']]

            if track['album']['id'] not in albums.index:
                printlog(
                    str(track['album']['id']) +
                    ' : ' + str(track['album']['name']) +
                    ' : ' + str(track['album']['release_date']) +
                    ' : ' + str(track['album']['release_date_precision'])
                )
                albums.loc[track['album']['id']] = [
                    artist_id, 
                    track['album']['name'], 
                    track['album']['release_date'], 
                    track['album']['release_date_precision']
                ]

            lyrics_list[track['id']] = lyrics
            previews[track['id']] = track['preview_url']
            if track['album']['id'] not in album_covers:
                try:
                    album_covers[track['album']['id']] = track['album']['images'][0]['url']
                except:
                    album_covers[track['album']['id']] = None
            elif album_covers[track['album']['id']] is None:
                try:
                    album_covers[track['album']['id']] = track['album']['images'][0]['url']
                except:
                    album_covers[track['album']['id']] = None
                

    for track in res_US['tracks']:
        if track['preview_url'] is None:
            break
                
        if track['id'] in previews:
            if previews[track['id']] is None:
                try:
                    lyrics = get_track_lyrics(track['name'], artist_name)
                except Exception:
                    continue

                if lyrics is not None:
                    printlog(
                        str(track['id']) +
                        ' : ' + str(track['name'])
                    )
                    tracks.loc[track['id']] = [artist_id, track['album']['id'], track['name']]

                    if track['album']['id'] not in albums.index:
                        printlog(
                            str(track['album']['id']) +
                            ' : ' + str(track['album']['name']) +
                            ' : ' + str(track['album']['release_date']) +
                            ' : ' + str(track['album']['release_date_precision'])
                        )
                        albums.loc[track['album']['id']] = [
                            artist_id, 
                            track['album']['name'], 
                            track['album']['release_date'], 
                            track['album']['release_date_precision']
                        ]

                    lyrics_list[track['id']] = lyrics
                    previews[track['id']] = track['preview_url']
                    if track['album']['id'] not in album_covers:
                        try:
                            album_covers[track['album']['id']] = track['album']['images'][0]['url']
                        except:
                            album_covers[track['album']['id']] = None
                    elif album_covers[track['album']['id']] is None:
                        try:
                            album_covers[track['album']['id']] = track['album']['images'][0]['url']
                        except:
                            album_covers[track['album']['id']] = None
        else:
            try:
                lyrics = get_track_lyrics(track['name'], artist_name)
            except Exception:
                continue

            if lyrics is not None:
                printlog(
                    str(track['id']) +
                    ' : ' + str(track['name'])
                )
                tracks.loc[track['id']] = [artist_id, track['album']['id'], track['name']]

                if track['album']['id'] not in albums.index:
                    printlog(
                        str(track['album']['id']) +
                        ' : ' + str(track['album']['name']) +
                        ' : ' + str(track['album']['release_date']) +
                        ' : ' + str(track['album']['release_date_precision'])
                    )
                    albums.loc[track['album']['id']] = [
                        artist_id, 
                        track['album']['name'], 
                        track['album']['release_date'], 
                        track['album']['release_date_precision']
                    ]

                lyrics_list[track['id']] = lyrics
                previews[track['id']] = track['preview_url']
                if track['album']['id'] not in album_covers:
                    try:
                        album_covers[track['album']['id']] = track['album']['images'][0]['url']
                    except:
                        album_covers[track['album']['id']] = None
                elif album_covers[track['album']['id']] is None:
                    try:
                        album_covers[track['album']['id']] = track['album']['images'][0]['url']
                    except:
                        album_covers[track['album']['id']] = None

    return tracks, albums, lyrics_list, previews, album_covers

def fr_get_all_tracks(country, artist_name, artist_id):
    # id: {artist_id: '', name: '', release_date: '', release_date_precision: ''}
    col =  ['artist_id', 'name', 'release_date', 'release_date_precision']
    albums = pd.DataFrame(columns=col)

    # id: {artist_id: '', album_id: '', name: ''}
    col =  ['artist_id', 'album_id', 'name']
    tracks = pd.DataFrame(columns=col)

    lyrics_list = {}
    previews = {}
    album_covers = {}

    result_albums = SPOTIFY.artist_albums(artist_id, limit=50)

    dups = re.compile(r'Deluxe|Remastered|Remaster|Live|Version|Edition')

    albums_markets = {}
    album_names = []
    for album in result_albums['items']:
        if not dups.search(album['name']) and album['name'].lower() not in album_names:
            if album['album_type'] in ['album', 'single']:
                markets = []
                if country in album['available_markets']:
                    markets.append(country)
                if 'US' in album['available_markets'] and country != 'US':
                    markets.append('US')
                markets.extend(random.choices(album['available_markets'], k=3-len(markets)))

                albums_markets[album['id']] = {
                    'markets': set(markets), 
                    'artist_id': album['artists'][0]['id'], 
                    'name': album['name'], 
                    'release_date': album['release_date'], 
                    'release_date_precision': album['release_date_precision'],
                    'img_url':album['images'][0]['url']
                }
                album_names.append(album['name'].lower())

    for album_id, album in albums_markets.items():
        printlog(f'Album: {album["name"]}')
        
        result_tracks_albums = SPOTIFY.album_tracks('6i6folBtxKV28WX3msQ4FE')
        tracks_without_previews = []
        for track in result_tracks_albums['items']:
            if track['id'] not in previews:
                if track['preview_url'] is None:
                    tracks_without_previews.append(track['id'])
                else:
                    try:
                        lyrics = get_track_lyrics(track['name'], artist_name)
                    except Exception:
                        continue

                    if lyrics is not None:
                        printlog(
                            str(track['id']) +
                            ' : ' + str(track['name'])
                        )
                        tracks.loc[track['id']] = [artist_id, album_id, track['name']]

                        if album_id not in albums.index:
                            printlog(
                                str(album_id) +
                                ' : ' + str(album['name']) +
                                ' : ' + str(album['release_date']) +
                                ' : ' + str(album['release_date_precision'])
                            )
                            albums.loc[album_id] = [
                                artist_id, 
                                album['name'], 
                                album['release_date'], 
                                album['release_date_precision']
                            ]

                        lyrics_list[track['id']] = lyrics
                        previews[track['id']] = track['preview_url']
                        album_covers[album_id] = album['img_url']

        for market in album['markets']:
            res_tracks = SPOTIFY.tracks(tracks_without_previews, market=market)
            for track in res_tracks['tracks']:
                if track['id'] not in previews and track['preview_url'] is not None:
                    try:
                        lyrics = get_track_lyrics(track['name'], artist_name)
                    except Exception:
                        continue

                    if lyrics is not None:
                        printlog(
                            str(track['id']) +
                            ' : ' + str(track['name'])
                        )
                        tracks.loc[track['id']] = [artist_id, album_id, track['name']]

                        if album_id not in albums.index:
                            printlog(
                                str(album_id) +
                                ' : ' + str(album['name']) +
                                ' : ' + str(album['release_date']) +
                                ' : ' + str(album['release_date_precision'])
                            )
                            albums.loc[album_id] = [
                                artist_id, 
                                album['name'], 
                                album['release_date'], 
                                album['release_date_precision']
                            ]

                        lyrics_list[track['id']] = lyrics
                        previews[track['id']] = track['preview_url']
                        album_covers[album_id] = album['img_url']
    if len(tracks) == 0:
        return fr_get_top_tracks_albums(country, artist_name, artist_id)
    else:
        return tracks, albums, lyrics_list, previews, album_covers 

####### Location #######
def get_lat_long(location):
    printlog(f'get_lat_long({location})')
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?address='
    key = '&key=' + MAPS_API_KEY
    response = requests.get(base_url + location + key)
    resp_json_payload = response.json()
    results = resp_json_payload['results'][0]
    country = None
    for comp in results['address_components']:
        if 'country' in comp['types']:
            country = pycountry.countries.lookup(comp['long_name']).alpha_2
    return results['geometry']['location'], country
    
def clean_wiki_location_text(location):
    months = re.compile(
        '(January)|(February)|(March)|(April)|(May)|(June)|(July)|(August)|(September)|(October)|(November)|(December)',
        re.IGNORECASE
    )
    location = re.sub(months, '', location, re.IGNORECASE)
    location = re.sub(re.compile('Born', re.IGNORECASE), '', location, re.IGNORECASE)
    location = re.sub(re.compile('Age', re.IGNORECASE), '', location, re.IGNORECASE)
    location = re.sub(re.compile('Origin', re.IGNORECASE), '', location, re.IGNORECASE)
    location = re.sub(re.compile('[0-9]'), '', location)
    location = re.sub(re.compile('\\W'), '', location)
    return location

def get_metadata_wiki(artist_name):
    req = requests.get(wikipedia.page(artist_name).url)  
    b = BeautifulSoup(req.text, features='lxml')

    mb_id, origin, born = None, None, None
    
    for match in b.find_all(href=re.compile('https://musicbrainz.org/artist/\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12}')):
        mb_id = re.search('\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12}', str(match))[0]
        
    for table in b.find_all('table'):
        for row in table.find_all('tr'):
            if re.search('>born<', str(row), re.IGNORECASE): 
                born = row.get_text()
            if re.search('>origin<', str(row), re.IGNORECASE): 
                origin = row.get_text()

    if born is not None:
        born = clean_wiki_location_text(born)
    if origin is not None:
        origin = clean_wiki_location_text(origin)
            
    printlog(f'get_metadata_wiki({artist_name}) results: mb_id : {mb_id} born : {born} origin : {origin}')

    return mb_id, born, origin
   
def get_metadata_mm(artist_name):
    base_url = "https://api.musixmatch.com/ws/1.1/" + "artist.search" + "?format=json&callback=callback"
    artist_search = "&q_artist=" + artist_name
    api = "&apikey=" + MM_API_KEY
    api_call = base_url + artist_search + '&page_size=1' + api
    request = requests.get(api_call)
    result = request.json()
    
    mm_id, country = None, None
    
    for artist in result['message']['body']['artist_list']:
        if artist['artist']['artist_name'].lower() == artist_name.lower():
            prtstr = ''
            if 'artist_id' in artist['artist']:
                mm_id = artist['artist']['artist_id']
                prtstr = prtstr + str(mm_id) + ' : '
            if 'artist_country' in artist['artist']:
                country = str(artist['artist']['artist_country'])
                prtstr = prtstr + country
            printlog(f'get_metadata_mm({artist_name}) results: {prtstr}')
    
    try:
        country = pycountry.countries.lookup(country).name
    except Exception: 
        pass
    
    return mm_id, country

def get_metadata_mb(artist_name):
    result = musicbrainzngs.search_artists(artist=artist_name)

    mb_id, area1, area2, area3 = None, '', '', ''

    for artist in result['artist-list']:
        if artist['name'].lower() == artist_name.lower():
            prtstr = ''
            if 'id' in artist:
                mb_id = artist['id']
                prtstr = prtstr + artist['id'] + ' : '
            if 'begin-area' in artist:
                if 'name' in artist['begin-area']:
                    area1 = str(artist['begin-area']['name'])
                    prtstr = prtstr + area1 + ' : '
            if 'area' in artist:
                if 'name' in artist['area']:
                    area2 = str(artist['area']['name'])
                    prtstr = prtstr + area2 + ' : ' 
            if 'country' in artist:
                area3 = str(artist['country'])
                prtstr = prtstr + area3 + ' : ' 
            printlog(f'get_metadata_mb({artist_name}) results: {prtstr}')
            break

    try:
        area3 = pycountry.countries.lookup(area3).name
    except Exception: 
        pass

    return mb_id, area1, area2, area3

def get_metadata_mb_id(mb_id, artist_name):
    result = musicbrainzngs.get_artist_by_id(mb_id)
    
    mb_id, area1, area2, area3 = None, '', '', ''
    
    artist = result['artist']    

    if artist['name'].lower() == artist_name.lower():
        prtstr = ''
        if 'id' in artist:
            mb_id = artist['id']
            prtstr = prtstr + artist['id'] + ' : '
        if 'name' in artist['begin-area']:
            area1 = str(artist['begin-area']['name'])
            prtstr = prtstr + area1 + ' : '
        if 'name' in artist['area']:
            area2 = str(artist['area']['name'])
            prtstr = prtstr + area2 + ' : ' 
        if 'country' in artist:
            area3 = str(artist['country'])
            prtstr = prtstr + area3 + ' : ' 
        printlog(f'get_metadata_mb_id({mb_id}, {artist_name}) results: {prtstr}')
            
    try:
        area3 = pycountry.countries.lookup(area3).name
    except Exception: 
        pass
        
    return mb_id, area1, area2, area3

def get_artist_location(artist_name):
    mb, wiki, mm = False, False, False
    mb_wid, area1, area2 = None, None, None
    area3, origin, born, country, location = None, None, None, None, None 
    
    printlog('Try MusicBrainz...')
    try:
        _, area1, area2, area3 = get_metadata_mb(artist_name)
        mb = True
    except Exception:
        printlog('MusicBrainz entry not found, try Wikipedia...', e=True)
        try:
            mb_wid, origin, born = get_metadata_wiki(artist_name)
            wiki = True
        except Exception:
            printlog('Wikipedia entry not found, try Musixmatch...', e=True)
            try:
                mm_id, country = get_metadata_mm(artist_name)
                mm = True
            except Exception:
                printlog('Nothing found, exit', e=True)
                return location, country
    
    if mb:
        printlog('Try to get the location from MusicBrainz...')
        try:
            location, country = get_lat_long(area1 + '+' + area2 + '+' + area3)
        except Exception:
            printlog('Failed to get the location from MusicBrainz, try Wikipedia...', e=True)
            try:
                mb_wid, origin, born = get_metadata_wiki(artist_name)
                if origin is not None:
                    location, country = get_lat_long(origin)
                elif born is not None:
                    location, country = get_lat_long(born)
            except Exception:
                try:
                    if mb_wid is not None:
                        printlog('Failed to get the location from Wikipedia, try MusicBrainz again with wiki mb_id...', e=True)
                        mb_id, area1, area2, area3 = get_metadata_mb_id(mb_wid, artist_name)
                        location, country = get_lat_long(area1 + '+' + area2 + '+' + area3)
                    else:
                        raise Exception('mb_wid is None')
                except Exception:
                    printlog('Failed to get the location from MusicBrainz with wiki mb_id, try Musixmatch...', e=True)
                    try:
                        mm_id, country = get_metadata_mm(artist_name)
                        location, country = get_lat_long(country)
                    except Exception:
                        printlog('Location not found, exit', e=True)
                        return location, country
                
    if wiki:
        printlog('Try to get the location from Wikipedia...')
        try:
            if origin is not None:
                location, country = get_lat_long(origin)
            elif born is not None:
                location, country = get_lat_long(born)
        except Exception:
            try:
                if mb_wid is not None:
                    printlog('Failed to get the location from Wikipedia, try MusicBrainz again with wiki mb_id...', e=True)
                    mb_id, area1, area2, area3 = get_metadata_mb_id(mb_wid, artist_name)
                    location, country = get_lat_long(area1 + '+' + area2 + '+' + area3)
                else:
                    raise Exception('mb_wid is None')
            except Exception:
                printlog('Failed to get the location from MusicBrainz with wiki mb_id, try Musixmatch...', e=True)
                try:
                    mm_id, country = get_metadata_mm(artist_name)
                    location, country = get_lat_long(country)
                except Exception:
                    printlog('Location not found, exit', e=True)
                    return location, country
                
    if mm:
        printlog('Try to get the location from Musixmatch...')
        try:
            mm_id, country = get_metadata_mm(artist_name)
            location, country = get_lat_long(country)
        except Exception:
            printlog('Location not found, exit', e=True)
            return location, country                 
        
    return location, country


if __name__=='__main__':
    print('File Start...')
    file_start = time.perf_counter()

    ARTISTS, FUTURE_ARTISTS, SEED_ARTISTS, ALBUMS, TRACKS = get_dataframes()
    backup_dataframes()

    if args.print_top_genres:
        top_genres = [
            'classic rock', 'alternative rock', 'indie rock', 'psychedelic rock', 'rock', 'dance pop', 'pop', 
            'punk', 'experimental', 'soul', 'electronic', 'funk', 'r&b', 'christmas', 'hip hop', 'rap', 
            'rockabilly', 'grunge', 'industrial', 'reggae', 'country', 'blues', 'emo', 'video game music',
            'ska', 'jazz', 'disco', 'edm', 'doo-wop', 'lo-fi', 'metal', 'folk', 'soundtrack'
        ]
        print_top_genres(df=ARTISTS, top_genres=top_genres)

    if args.seeds_reset:
        printlog(f'Marking seeds that failed as unscraped....')
        try:
            reset_failed_seed_artists(seed=SEED_ARTISTS, artists=ARTISTS)
            save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)
        except Exception:
            printlog(f'Exception occured, could not mark failed seeds as unscraped', e=True)

    if args.set_seed_unscraped is not None:
        printlog(f'Marking seed {args.set_seed_unscraped} as unscraped....')
        try:
            mark_seed_as_unscraped(df=SEED_ARTISTS, seed_artist_id=args.set_seed_unscraped)
            save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)
        except Exception:
            printlog(f'Exception occured, could not mark seed as unscraped', e=True)


    if args.seeds is not None:
        printlog(f'Adding {args.seeds} to seed_artists list...')
        try:
            SEED_ARTISTS = inject_seed_artists(df=SEED_ARTISTS, list_ids=args.seeds.split(','), top=args.seeds_top)
            save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)
        except Exception:
            printlog(f'Exception occured, could not add seeds {args.seeds}', e=True)

    num_seed_artists = int(args.num_seed_artists)

    while num_seed_artists > 0:
        ################## Get next seed artist #########################################
        try:
            printlog('Getting next artist from seed_artists...')

            seed_artist_id = get_next_seed_artist(seed_artists=SEED_ARTISTS)
            printlog(f'{seed_artist_id} obtained as the next seed artist')

            if seed_artist_id == -1:
                printlog('All seed_artists have been scraped, add more seed artists!')
                break

        except Exception:
            printlog('Exception occured getting next artist!', e=True)
            break

        ################## Get related artists ##########################################
        try:
            # id: '', name: '', genres: [], lat: ?, lng: ?
            printlog(f'Getting related artists...')

            related_artists = fr_get_related(res=SPOTIFY.artist_related_artists(seed_artist_id))

            printlog(f'Success getting related artists.')

        except Exception:
            printlog('Exception occured getting related artists!', e=True)
            break

        ################## Add related artists to seed artists ##########################
        try:
            printlog(f'Adding related artists to seed_artists list...')

            inject_seed_artists(df=SEED_ARTISTS, list_ids=related_artists.index.values)

            printlog(f'Related added to seed_artists list.')

            save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)

        except Exception:
            printlog(f'Exception occured adding related artists to seeds!', e=True)
            break

        ################## Mark seed artist as scraped ##################################
        try:
            printlog(f'Mark {seed_artist_id} as scraped...')

            mark_seed_as_scraped(df=SEED_ARTISTS, seed_artist_id=seed_artist_id)

            printlog(f'{seed_artist_id} marked as scraped.')

            save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)

        except Exception:
            printlog(f'Exception occured marking {seed_artist_id} as scraped!', e=True)
            break

        ################## Add related artist metadata to future artists ################
        try:
            printlog(f'Adding related artists metadata to future artists...')

            for index, row in related_artists.iterrows():
                add_artist(df=FUTURE_ARTISTS, artist_id=index, name=row['name'], genres=row['genres'], related=None, lat=None, lng=None)

            printlog(f'Success adding related artists metadata to future artists.')

            save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)

        except Exception:
            printlog(f'Exception occured adding related artists metadata to future artists.!', e=True)
            break

        ################## Get metadata for seed artist #################################
        try:
            # id: '', name: '', genres: [], lat: ?, lng: ?
            printlog(f'Getting metadata for seed artist with ID: {seed_artist_id}...') 

            printlog(f'Check if its been loaded into future artists before...') 

            if seed_artist_id in FUTURE_ARTISTS.index:
                printlog(f'Getting metadata from future artists...') 
                artist_metadata = FUTURE_ARTISTS.loc[[seed_artist_id]]

            elif seed_artist_id in ARTISTS.index:
                raise Exception("Seed artist already saved as an artist. This should not happen!")

            else:
                printlog(f'Not found in future artists, get from spotify...') 
                artist_metadata = fr_get_artist_metadata(res=SPOTIFY.artist(seed_artist_id))

            printlog(f'Success getting metadata!') 

        except Exception:
            printlog('Exception occured getting metadata!', e=True)
            break

        ################## Get location #################################################
        try:
            # {lat: 0.0, lng: 0.0}
            printlog(f'Getting location of artist...')

            location, country = get_artist_location(artist_metadata.loc[seed_artist_id]['name'])

            if location is None:
                printlog(f'Location could not be found! Bummer, cant use this artist. Try next seed artist...')
                num_seed_artists -= 1
                continue

            artist_metadata.loc[seed_artist_id]['lat'] = location['lat']
            artist_metadata.loc[seed_artist_id]['lng'] = location['lng']

            printlog(f'Success getting location.')

        except Exception:
            printlog(f'Exception occured getting location!', e=True)
            break
    
        ################## Get top tracks and albums ####################################
        try:
            # id: {artist_id: '', album_id: '', name: ''}
            # id: {artist_id: '', name: '', release_date: '', release_date_precision: ''}

            # id: lyric string
            # id: preview url (.mp3)
            # id: album cover ulr (.jpg)

            printlog(f'Getting top tracks for seed artist...')

            tracks, albums, lyrics, previews, album_covers = fr_get_top_tracks_albums(
                country=country, 
                artist_name=artist_metadata.loc[seed_artist_id]['name'], 
                artist_id=seed_artist_id
            )    

            if len(tracks) == 0:
                printlog(f'No tracks found! Bummer, cant use this artist. Try next seed artist...')
                num_seed_artists -= 1
                printlog('continue')

        except Exception:
            printlog('Exception occured getting tracks and albums!', e=True)
            break

        ################## Download album art ###########################################
        albums_downloaded = set()
        for i, url in album_covers.items():
            try:
                printlog(f'Downloading album {i}...')
                urllib.request.urlretrieve(url, './albums/temp/' + i + '.jpg') 
                albums_downloaded.add(i)
                printlog(f'Done!')
            except Exception:
                printlog(f'Error downloading album {url}', e=True)

        ################## Download track audio clip ####################################
        tracks_downloaded = set()
        for i, url in previews.items():
            if tracks.loc[i]['album_id'] in albums_downloaded:
                try:
                    printlog(f'Downloading track {i}...')
                    urllib.request.urlretrieve(url, './audio/temp/' + i + '.mp3') 
                    tracks_downloaded.add(i)
                    printlog(f'Done!')
                except Exception:
                    printlog(f'Error downloading track {url}', e=True)

        ################## Save lyrics to file ##########################################
        lyrics_saved = set()
        for i, lyric in lyrics.items():
            if i in tracks_downloaded:
                try:
                    printlog(f'Saving lyric {i}...')
                    with open('./lyrics/temp/' + i + '.txt', 'w') as text_file:
                        print(f"{lyric}", file=text_file)
                    lyrics_saved.add(i)
                    printlog(f'Done!')
                except Exception:
                    printlog(f'Error saving lyric {lyric}', e=True)             

        ################## Save to data file ############################################
        final_tracks = set()
        final_albums = set()
        for i in tracks_downloaded:
            try:
                assert(i not in TRACKS.index), 'Track already in tracks'
                assert(tracks.loc[i]['artist_id'] not in ARTISTS.index), 'Track artist already in artists'
                assert(tracks.loc[i]['album_id'] not in ALBUMS.index), 'Track album already in albums'
                assert(tracks.loc[i]['artist_id'] == seed_artist_id), 'Track artist does not match the seed artist'
                assert(tracks.loc[i]['artist_id'] in artist_metadata.index), 'Track artist does not match the seed artist metadata'
                assert(tracks.loc[i]['album_id'] in albums_downloaded), 'Track album was not downloaded'
                assert(i in lyrics_saved), 'Track lyrics were not saved'
                final_tracks.add(i)
                final_albums.add(tracks.loc[i]['album_id'])
            except AssertionError:
                printlog(f'Failed to add track {i}', e=True)
            
        if len(final_tracks) > 0:
            try:
                printlog('Saving scraped data to the dataframes...')

                for i in final_tracks:
                    add_track(
                        df=TRACKS, 
                        track_id=i, 
                        artist_id=seed_artist_id, 
                        album_id=tracks.loc[i]['album_id'], 
                        name=tracks.loc[i]['name'], 
                    )
                    
                for i in final_albums:
                    add_albums(
                        df=ALBUMS, 
                        album_id=i,
                        artist_id=seed_artist_id, 
                        name=albums.loc[i]['name'], 
                        release_date=albums.loc[i]['release_date'],
                        release_date_precision=albums.loc[i]['release_date_precision']
                    )

                add_artist(
                    df=ARTISTS, 
                    artist_id=seed_artist_id,
                    name=artist_metadata.loc[seed_artist_id]['name'], 
                    genres=artist_metadata.loc[seed_artist_id]['genres'], 
                    related=related_artists.index.values,
                    lat=artist_metadata.loc[seed_artist_id]['lat'], 
                    lng=artist_metadata.loc[seed_artist_id]['lng']
                )

                save_dataframes(artists=ARTISTS, future_artists=FUTURE_ARTISTS, seed_artists=SEED_ARTISTS, albums=ALBUMS, tracks=TRACKS)

                ################## Move temp files out of temp ##################################
                time.sleep(5)
                try:
                    printlog('Moving temp files into the permanent location...')

                    for i in final_tracks:
                        os.rename('./lyrics/temp/' + i + '.txt', './lyrics/' + i + '.txt')
                        os.rename('./audio/temp/' + i + '.mp3', './audio/' + i + '.mp3')
                    
                    for i in final_albums:
                        os.rename('./albums/temp/' + i + '.jpg', './albums/' + i + '.jpg')

                    for i in albums_downloaded:
                        if i not in final_albums:
                            os.remove('./albums/temp/' + i + '.jpg')

                    for i in tracks_downloaded:
                        if i not in final_tracks:   
                            os.remove('./lyrics/temp/' + i + '.txt')
                            os.remove('./audio/temp/' + i + '.mp3') 


                    printlog('*' * 117)
                    printlog(f'Artist {seed_artist_id} added!')
                    printlog('*' * 117)

                except Exception:
                    printlog(f'Error moving files! Please move these manually {final_tracks}, {final_albums}', e=True)
                    break 

            except Exception:
                printlog('Error adding data to dataframes!', e=True)
                break 
        else:
            printlog('Failed to add any tracks! Bummer, cant use this artist. Try next seed artist...', e=True)
            num_seed_artists -= 1
            continue

        ################## Check integrity of the data ##################################
        integrityIsGood = True
        if (set([n[9:-4] for n in glob.glob("./albums/*.jpg")]) != set(ALBUMS.index.values)):
            printlog('Album covers saved saved to file do not match albums saved to data:')
            printlog(set([n[9:-4] for n in glob.glob("./albums/*.jpg")]).symmetric_difference(set(ALBUMS.index.values)))
            integrityIsGood = False
            
        if (set([n[8:-4] for n in glob.glob("./audio/*.mp3")]) != set(TRACKS.index.values)):
            printlog('Audio saved saved to file do not match tracks saved to data:')
            printlog(set([n[8:-4] for n in glob.glob("./audio/*.mp3")]).symmetric_difference(set(TRACKS.index.values)))
            integrityIsGood = False
            
        if (set([n[9:-4] for n in glob.glob("./lyrics/*.txt")]) != set(TRACKS.index.values)):
            printlog('Lyrics saved saved to file do not match tracks saved to data:')
            printlog(set([n[9:-4] for n in glob.glob("./lyrics/*.txt")]).symmetric_difference(set(TRACKS.index.values)))
            integrityIsGood = False
            
        if (set(TRACKS.artist_id.values) != set(ARTISTS.index.values)):
            printlog('Some tracks point to artists that dont exist:')
            printlog(set(TRACKS.artist_id.values).symmetric_difference(set(ARTISTS.index.values)))
            integrityIsGood = False

        if (set(TRACKS.album_id.values) != set(ALBUMS.index.values)):
            printlog('Some tracks point to albums that dont exist:')
            printlog(set(TRACKS.album_id.values).symmetric_difference(set(ALBUMS.index.values)))
            integrityIsGood = False

        if integrityIsGood:
            num_seed_artists -= 1
        else:
            break