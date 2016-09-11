# Parts of the code are taken from the MIT licensed repository:
# https://github.com/egbertbouman/youtube-comment-downloader

import os
import sys
import time
import json
import requests
import argparse
import lxml.html
import csv

from lxml.cssselect import CSSSelector
from time import sleep

YOUTUBE_COMMENTS_URL = 'https://www.youtube.com/all_comments?v={youtube_id}'
YOUTUBE_COMMENTS_AJAX_URL = 'https://www.youtube.com/comment_ajax'

USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'


def non_private_videos(idlist, pbar=None):
    """
    Checks which videos in the idlist are non private and pbar is a progress bar
    implementation
    """
    videos = []
    update = lambda x: None
    if (not (pbar is None)) and callable(pbar.update):
        update = pbar.update

    for video_id in idlist:
        if not is_video_private(video_id):
            videos += [video_id]
        update(1)
        sleep(0.5)
    return videos

def is_video_private(youtube_id):
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT

    # Get Youtube page with initial comments
    response = session.get(YOUTUBE_COMMENTS_URL.format(youtube_id=youtube_id))
    html = response.text
    # Check if video is private
    return ("video is private" in html) or ("Sorry about that." in html)

def unique_videos_in_dataset(file):
    with open(file) as csvfile:
        encountered = {}
        videos = []
        youtube_id = csv.reader(csvfile, delimiter = ',')
        unique = 0
        for row in youtube_id:
            if row[0] not in encountered:
                encountered[row[0]] = True
                videos += [row[0]]
                unique += 1
            if row[1] not in encountered:
                encountered[row[1]] = True
                videos += [row[1]]
                unique += 1
        return videos

def find_value(html, key, num_chars=2):
    pos_begin = html.find(key) + len(key) + num_chars
    pos_end = html.find('"', pos_begin)
    return html[pos_begin: pos_end]


def extract_comments(html):
    tree = ""
    try:
        tree = lxml.html.fromstring(html)
    except Exception:
        return {'cid': 'XXXX',
               'text': 'XXXX',
               'time': 'XXXX',
               'author': 'XXXX'}
        pass
        # Log error
    item_sel = CSSSelector('.comment-item')
    text_sel = CSSSelector('.comment-text-content')
    time_sel = CSSSelector('.time')
    author_sel = CSSSelector('.user-name')

    for item in item_sel(tree):
        yield {'cid': item.get('data-cid'),
               'text': text_sel(item)[0].text_content(),
               'time': time_sel(item)[0].text_content().strip(),
               'author': author_sel(item)[0].text_content()}


def extract_reply_cids(html):
    tree = ""
    try:
        tree = lxml.html.fromstring(html)
    except Exception as err:
        raise
        # TODO: log these errors
        # print("Error: {0} ".format(err) + " while opening: " + html)
    sel = CSSSelector('.comment-replies-header > .load-comments')
    return [i.get('data-cid') for i in sel(tree)]


def ajax_request(session, url, params, data, retries=10, sleep=20):
    for _ in range(retries):
        response = session.post(url, params=params, data=data)
        if response.status_code == 200:
            response_dict = json.loads(response.text)
            return response_dict.get('page_token', None), response_dict['html_content']
        else:
            time.sleep(sleep)


def download_comments(youtube_id, sleep=1):
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT

    # Get Youtube page with initial comments
    response = session.get(YOUTUBE_COMMENTS_URL.format(youtube_id=youtube_id))
    html = response.text
    # Check if video is private
    if "video is private" in html:
        return {'cid': 'XXXX',
               'text': 'XXXX',
               'time': 'XXXX',
               'author': 'XXXX'}

    reply_cids = 'XXXX'
    try:
        reply_cids = extract_reply_cids(html)
    except:
        pass

    ret_cids = []
    for comment in extract_comments(html):
        ret_cids.append(comment['cid'])
        yield comment

    page_token = find_value(html, 'data-token')
    session_token = find_value(html, 'XSRF_TOKEN', 4)

    first_iteration = True

    # Get remaining comments (the same as pressing the 'Show more' button)
    while page_token:
        data = {'video_id': youtube_id,
                'session_token': session_token}

        params = {'action_load_comments': 1,
                  'order_by_time': True,
                  'filter': youtube_id}

        if first_iteration:
            params['order_menu'] = True
        else:
            data['page_token'] = page_token

        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL, params, data)
        if not response:
            break

        page_token, html = response

        try:
            reply_cids += extract_reply_cids(html)
        except:
            reply_cids += 'XXXX'
            pass
        for comment in extract_comments(html):
            if comment['cid'] not in ret_cids:
                ret_cids.append(comment['cid'])
                yield comment

        first_iteration = False
        time.sleep(sleep)

    # Get replies (the same as pressing the 'View all X replies' link)
    for cid in reply_cids:
        data = {'comment_id': cid,
                'video_id': youtube_id,
                'can_reply': 1,
                'session_token': session_token}

        params = {'action_load_replies': 1,
                  'order_by_time': True,
                  'filter': youtube_id,
                  'tab': 'inbox'}

        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL, params, data)
        if not response:
            break

        _, html = response

        for comment in extract_comments(html):
            if comment['cid'] not in ret_cids:
                ret_cids.append(comment['cid'])
                yield comment
        time.sleep(sleep)