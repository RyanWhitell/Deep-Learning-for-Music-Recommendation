import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import wikipedia
import musicbrainzngs

import urllib.request
import requests
from bs4 import BeautifulSoup
import re

import h5py
import time
import datetime
import pandas as pd
import pycountry

from shutil import copyfile
import logging
import argparse

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

musicbrainzngs.set_useragent('haamr', 1.0)
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
SPOTIFY = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

parser = argparse.ArgumentParser(description="scrapes various apis for music content")
parser.add_argument('-n', '--num-seed-artists', default=0, help='number of seed_artists to scrape')
parser.add_argument('-s', '--seeds', default=None, help='injects seed artists via comma separated list')
args = parser.parse_args()

####### Utility #######
def printlog(message, e=False):
    print(message)
    if e:
        logging.exception(message)
    else:
        logging.info(message)

####### Data #######
def inject_seed_artists(list_ids):
    f = h5py.File('./data.hdf5', 'r+')

    seed_artists = f['seed_artists']

    added = []
    for i in list_ids:
        if i in seed_artists:
            printlog(f'id {i} already in seed_artists')
        else:
            seed_artists[i] = False
            added.append(i)

    f.close()
    return added

def get_next_seed_artist():
    f = h5py.File('./data.hdf5', 'r')

    seed_artists = f['seed_artists']

    artist_id = -1
    for i in seed_artists.keys():
        if not seed_artists[i][()]:
            artist_id = i
            break

    f.close()
    return artist_id

def mark_seed_as_scraped(seed_artist_id):
    f = h5py.File('./data.hdf5', 'r+')

    seed_artists = f['seed_artists']

    if seed_artist_id in seed_artists:
        seed_artists[seed_artist_id][()] = True

    f.close()

def save_artist(artist_id, name, genres, followers, popularity, future):
    f = h5py.File('./data.hdf5', 'r+')

    if future:
        artists = f['future_artists']
    else:
        artists = f['artists']

    artist_group = artists.create_group(artist_id)
    artist_info = artist_group.create_dataset('info', (3,), dtype=str, data=[name, followers, popularity])
    artist_genres = artist_group.create_dataset('info', (len(genres),), dtype=str, data=genres)

def save_track(track_dict):

def save_album(album_dict):
####### Artist #######
def fr_get_related(res):
    
    related_artists_ids = []

    for artist in res['artists']:
        related_artists.append(artist['id'])
    return related_artists_ids

def fr_get_artist_metadata(res):
    f = h5py.File('./data.hdf5', 'r')
    
    artists = f['artists']

    artist_dict = None

    if res['id'] not in artists:
        printlog(
            str(res['id']) +
            ' : ' + str(res['name']) +
            ' : ' + str(res['genres']) +
            ' : ' + str(res['followers']['total']) +
            ' : ' + str(res['popularity'])
        )
        artist_dict = {
            'id' : res['id'],
            'name' : res['name'],
            'genres' : res['genres'],
            'followers' : res['followers']['total'],
            'popularity' : res['popularity']
        }
        
    f.close()
    return artist_dict

####### Track #######
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
        lyrics = re.sub(re.compile('\\[Verse \\d\\]|\\[Verse\\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile('\\[Chorus \\d\\]|\\[Chorus\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile('\\[Bridge \\d\\]|\\[Bridge\\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile('\\[Outro \\d\\]|\\[Outro\\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile('\\[Instrumental \\d\\]|\\[Instrumental\\]', re.IGNORECASE), '', lyrics)
        lyrics = re.sub(re.compile('^(\\n)*|(\\n)*$'), '', lyrics)
        return lyrics
    else:
        raise Exception(f'Artist not found, no hit from {hits} matches {artist_name.lower()}')
    
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

def get_track_lyrics(song_title, artist_name):
    lyrics = None
    
    printlog('Try Genius...')
    try:
        lyrics = get_lyrics_genius(song_title, artist_name)
    except Exception:
        printlog('Genius lyrics not found, try AZ...', e=True)
        try:
            lyrics = get_lyrics_az(song_title, artist_name)
        except Exception:
            printlog('No lyrics found, exit', e=True)
            return lyrics
        
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
    return data['lyrics']['lyrics_body']
  
def fr_get_top_tracks_albums(res, artist_name):
    tracks_dict = {}
    album_dict = {}

    for track in res['tracks']:
        try:
            lyrics = get_track_lyrics(track['name'], artist_name)
        except Exception:
            continue
            
        if lyrics is not None:
            printlog(
                str(track['id']) +
                ' : ' + str(track['external_ids']) +
                ' : ' + str(track['name']) +
                ' : ' + str(track['popularity']) +
                ' : ' + str(track['preview_url']) +
                ' : ' + str(track['album']['id'])
            )
            temp_dict = {
                'external_ids' : track['external_ids'],
                'name' : track['name'],
                'popularity' : track['popularity'],
                'preview_url' : track['preview_url'],
                'album_id' : track['album']['id'],
                'lyrics' : lyrics
            }
            tracks_dict[track['id']] = temp_dict

            if track['album']['id'] not in album_dict:
                printlog(
                    ' : ' + str(track['album']['id']) +
                    ' : ' + str(track['album']['images'][0]['url']) +
                    ' : ' + str(track['album']['name']) +
                    ' : ' + str(track['album']['release_date']) +
                    ' : ' + str(track['album']['release_date_precision'])
                )
                temp_dict = {
                    'image_url' : track['album']['images'][0]['url'],
                    'name' : track['album']['name'],
                    'release_date' : track['album']['release_date'],
                    'release_date_precision' : track['album']['release_date_precision']
                }
                album_dict[track['album']['id']] = temp_dict

    return tracks_dict, album_dict

####### Location #######
def get_lat_long(location):
    printlog(f'get_lat_long({location})')
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?address='
    key = '&key=' + MAPS_API_KEY
    response = requests.get(base_url + location + key)
    resp_json_payload = response.json()
    return resp_json_payload['results'][0]['geometry']['location']
  
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
    b = BeautifulSoup(req.text, 'lxml')

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
            if 'name' in artist['begin-area']:
                area1 = str(artist['begin-area']['name'])
                prtstr = prtstr + area1 + ' : '
            if 'name' in artist['area']:
                area2 = str(artist['area']['name'])
                prtstr = prtstr + area2 + ' : ' 
            if 'country' in artist:
                area3 = str(artist['country'])
                prtstr = prtstr + area3 + ' : ' 
            printlog(f'get_metadata_mb({artist_name}) results: {prtstr}')
            
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
    mb_id, mb_wid, mm_id, area1, area2 = None, None, None, None, None  
    area3, origin, born, country, location = None, None, None, None, None 
    
    printlog('Try MusicBrainz...')
    try:
        mb_id, area1, area2, area3 = get_metadata_mb(artist_name)
        mb = True
    except Exception:
        printlog('MusicBrainz entry not found, try Wikipedia...', e=True)
        try:
            mb_wid, origin, born = get_metadata_wiki(artist_name)
            wiki = True
        except Exception:
            print('Wikipedia entry not found, try Musixmatch...', e=True)
            try:
                mm_id, country = get_metadata_mm(artist_name)
                mm = True
            except Exception:
                printlog('Nothing found, exit', e=True)
                return mb_id, mb_wid, mm_id, location
    
    if mb:
        printlog('Try to get the location from MusicBrainz...')
        try:
            location = get_lat_long(area1 + '+' + area2 + '+' + area3)
        except Exception:
            printlog('Failed to get the location from MusicBrainz, try Wikipedia...', e=True)
            try:
                mb_wid, origin, born = get_metadata_wiki(artist_name)
                if origin is not None:
                    location = get_lat_long(origin)
                elif born is not None:
                    location = get_lat_long(born)
            except Exception:
                try:
                    if mb_wid is not None:
                        printlog('Failed to get the location from Wikipedia, try MusicBrainz again with wiki mb_id...', e=True)
                        mb_id, area1, area2, area3 = get_metadata_mb_id(mb_wid, artist_name)
                        location = get_lat_long(area1 + '+' + area2 + '+' + area3)
                    else:
                        raise Exception('mb_wid is None')
                except Exception:
                    printlog('Failed to get the location from MusicBrainz with wiki mb_id, try Musixmatch...', e=True)
                    try:
                        mm_id, country = get_metadata_mm(artist_name)
                        location = get_lat_long(country)
                    except Exception:
                        printlog('Location not found, exit', e=True)
                        return mb_id, mb_wid, mm_id, location
                
    if wiki:
        printlog('Try to get the location from Wikipedia...')
        try:
            if origin is not None:
                location = get_lat_long(origin)
            elif born is not None:
                location = get_lat_long(born)
        except Exception:
            try:
                if mb_wid is not None:
                    printlog('Failed to get the location from Wikipedia, try MusicBrainz again with wiki mb_id...', e=True)
                    mb_id, area1, area2, area3 = get_metadata_mb_id(mb_wid, artist_name)
                    location = get_lat_long(area1 + '+' + area2 + '+' + area3)
                else:
                    raise Exception('mb_wid is None')
            except Exception:
                printlog('Failed to get the location from MusicBrainz with wiki mb_id, try Musixmatch...', e=True)
                try:
                    mm_id, country = get_metadata_mm(artist_name)
                    location = get_lat_long(country)
                except Exception:
                    printlog('Location not found, exit', e=True)
                    return mb_id, mb_wid, mm_id, location
                
    if mm:
        printlog('Try to get the location from Musixmatch...')
        try:
            mm_id, country = get_metadata_mm(artist_name)
            location = get_lat_long(country)
        except Exception:
            printlog('Location not found, exit', e=True)
            return mb_id, mb_wid, mm_id, location                  
        
    return mb_id, mb_wid, mm_id, location


if __name__=='__main__':
    print('File Start...')
    file_start = time.perf_counter()

    now = datetime.datetime.now()
    now = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
    logging.basicConfig(filename='./dumps/' + now + '_.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    printlog('Backing up data file...')
    copyfile('./data.hdf5', './backups/' + now + '_' + 'data.hdf5')

    if args.seeds is not None:
        printlog(f'Adding {args.seeds} to seed_artists list...')
        try:
            added = inject_seed_artists(args.seeds.split(','))
            printlog(f'{added} added to seed_artists list')
        except Exception:
            printlog(f'Exception occured, could not add seeds {args.seeds}.hdf5', e=True)

    num_seed_artists = int(args.num_seed_artists)

    while num_seed_artists > 0:
        ################## Get next seed artist #########################################
        try:
            printlog('Getting next artist from seed_artists...')
            seed_artist_id = get_next_seed_artist()
            printlog(f'{seed_artist_id} obtained as the next seed artist')
            if seed_artist_id == -1:
                printlog('All seed_artists have been scraped, add more seed artists!')
                break
        except Exception:
            printlog('Exception occured getting next artist!', e=True)
            break

        ################## Get related artists ##########################################
        try:
            printlog(f'Getting related artists...')
            # [id, id, ...]
            related_artists = fr_get_related(SPOTIFY.artist_related_artists(seed_artist_id))
            if len(related_artists) == 0:
                printlog('This artist has no related artists :( Mark it as scraped and try the next one.')
                mark_seed_as_scraped(seed_artist_id)
                printlog(f'{seed_artist_id} marked as scraped!')
                num_seed_artists -= 1
                continue
            printlog(f'Success getting related artists: {related_artists}!')
        except Exception:
            printlog('Exception occured getting related artists!', e=True)
            break

        ################## Add related artists to seed artists ##########################
        try:
            printlog(f'Adding {related_artists} to seed_artists list...')
            added = inject_seed_artists(related_artists)
            printlog(f'{added} added to seed_artists list. Mark {seed_artist_id} as scraped...')
            mark_seed_as_scraped(seed_artist_id)
            printlog(f'{seed_artist_id} marked as scraped!')
        except Exception:
            printlog(f'Exception occured, could not add seeds {related_artists}.hdf5', e=True)
            break

        ################## Get metadata #################################################
        try:
            printlog(f'Getting metadata for seed artist with ID: {seed_artist_id}...') 
            # id: '', name: '', genres: [], followers: 0, popularity: 0
            artist_metadata = fr_get_artist_metadata(SPOTIFY.artist(seed_artist_id))
            if artist_metadata is None:
                printlog(f'Seed artist with ID: {seed_artist_id} has metadata already?')
                num_seed_artists -= 1
                continue
            printlog(f'Success getting metadata!') 
        except Exception:
            printlog('Exception occured getting metadata!', e=True)
            break

        ################## Get location #################################################
        try:
            printlog(f'Getting location of {artist_metadata["name"]}...')
            # {lat: 0.0, lng: 0.0}
            mb_id, mb_wid, mm_id, location = get_artist_location(artist_metadata['name'])
            if location is None:
                printlog(f'Location of {artist_metadata["name"]} could not be found...')
                num_seed_artists -= 1
                continue
            printlog(f'Success getting location: {location}')
        except Exception:
            printlog(f'Exception occured, could not get the location of {artist_metadata["name"]}.hdf5', e=True)
            break
    
        ################## Get top tracks and albums ####################################
        try:
            printlog(f'Getting top tracks for seed artist with ID: {seed_artist_id}...')
            # id: {external_ids: {'isrc': ''}, name: '', popularity: 0, preview_url: url, album_id: '', lyrics: ''}
            # id: {image_url: url, name: '', release_date: '', release_date_precision: ''}
            tracks, albums = fr_get_top_tracks_albums(SPOTIFY.artist_top_tracks(seed_artist_id), artist_metadata['name'])
            if len(tracks) == 0:
                printlog(f'No tracks found for {seed_artist_id}')
                num_seed_artists -= 1
                continue
        except Exception:
            printlog('Exception occured getting tracks and albums!', e=True)
            break

        ################## Download album art ###########################################
        albums_downloaded = set()
        for i, album in albums.items():
            try:
                printlog(f'Downloading album {i}...')
                urllib.request.urlretrieve(album['image_url'], './albums/temp/' + i + '.jpg') 
                albums_downloaded.add(i)
                printlog(f'Done!')
            except Exception:
                printlog(f'Error downloading album {i}', e=True)

        ################## Download track audio clip ####################################
        tracks_downloaded = set()
        for i, track in tracks.items():
            if track['album_id'] in albums_downloaded:
                try:
                    printlog(f'Downloading track {i}...')
                    urllib.request.urlretrieve(track['preview_url'], './audio/temp/' + i + '.mp3') 
                    tracks_downloaded.add(i)
                    printlog(f'Done!')
                except Exception:
                    printlog(f'Error downloading track {i}', e=True)

        ################## Save to file #################################################
        try:
            printlog(f'Saving all data...')
            f = h5py.File('./data.hdf5', 'r+')

            artists_group = f['artists']
            tracks_group = f['tracks']
            albums_group = f['albums']
            extras_group = f['extras']

            artists_group[artist_metadata['id']] = [artist_metadata['name'],  artist_metadata['genres'], artist_metadata['followers'], artist_metadata['popularity']]
            
            for i, track in tracks.items():
                if i in tracks_downloaded:
                    dataset = tracks_group.create_dataset(i, (5,), dtype=str)
                    for index, data in enumerate([artist_metadata['id'], track['album_id'], track['name'], track['popularity'], track['lyrics']]):
                        dataset[index] = data

            for i, album in albums.items():
                if i in albums_downloaded:
                    albums_group[i] = [artist_metadata['id'], album['name'], album['release_date'], album['release_date_precision']]

            for i, album in albums.items():
                if i in albums_downloaded:
                    dataset = albums_group.create_dataset(i, (4,), dtype=str)
                    for index, data in enumerate([artist_metadata['id'], album['name'], album['release_date'], album['release_date_precision']]):
                        dataset[index] = data

            f.close()
        except Exception:
            printlog(f'Could not save data!', e=True)
            break

        num_seed_artists -= 1