#!/usr/bin/env python3
"""
This module contains essential functions in order to download comments and replies to comments from youtube videos
"""
import argparse
import csv
import errno
import logging
import math
import os
import socket
import sys

from apiclient.discovery import build
from apiclient.errors import HttpError
from tqdm import tqdm


def get_comment_threads(youtube_service, video_id, csvfile, max_results=100, limit_pages=None, write_header=True):
    """
    Our purpose is to download comments and append them to a csv with the following format:
    videoId,textDisplay,isReplyTo,canReply,totalReplyCount,etag,id,authorChannelId,authorDisplayName,likeCount,
    publishedAt, updatedAt,viewerRating
    :param youtube_service: youtube object from the Youtube API, see get_default_youtube_service
    :param video_id: youtube video_id to download
    :param csvfile: the resulting csvfile to write
    :param max_results: max_results for each call (1-100)
    :param limit_pages: if set limits the comment pages to download, mainly for testing purposes
    :param write_header: write or not the csv header
    :return: top results but also write everything in the csvfile
    """

    logger = logging.getLogger("comment_error")
    logger.setLevel(logging.ERROR)
    file_handler = logging.FileHandler('youtube_comment_errors.log')
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    results = None

    # Setup csv file
    fieldnames = ['videoId', 'textDisplay', 'isReplyTo', 'canReply', 'totalReplyCount', 'kind', 'etag', 'id',
                  'authorChannelId', 'authorDisplayName', 'likeCount', 'publishedAt', 'updatedAt', 'viewerRating']
    with open(csvfile, 'a') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        def get_comment_page(page_results, ignore_first=False):
            """
            Processes a comment_page of results
            :param page_results: The results of a comment page
            :param ignore_first: Ignore the first result to avoid duplicates from previous calls
            :return:
            """
            first = True
            for item in page_results["items"]:
                if first:
                    first = False
                    if ignore_first:
                        continue
                process_comment(item)
                if 'replies' in item:
                    curitem = item["replies"]["comments"]
                    for comment_reply in curitem:
                        process_comment(comment_reply, is_reply=True,
                                        reply_to=item["snippet"]["topLevelComment"]["id"])

        def process_comment(item, is_reply=False, reply_to=None):
            """
            Processes one comment and writes the result to the csv file
            :param item: The comment to process
            :param is_reply: indicates if the comment is reply
            :param reply_to: the id of the comment it is a reply to
            :return:
            """
            snippet = item["snippet"]
            comment = snippet if is_reply else snippet["topLevelComment"]["snippet"]
            video_id2 = comment["videoId"]
            text_display = comment["textDisplay"]
            can_reply = snippet["canReply"] if "canReply" in comment else ""
            total_reply_count = snippet["totalReplyCount"] if "totalReplyCount" in snippet else 0
            kind = item["kind"]
            etag = snippet["topLevelComment"]["etag"] if not is_reply else item["etag"]
            id2 = snippet["topLevelComment"]["id"] if not is_reply else item["id"]
            author_channel_id = comment["authorChannelId"] if "authorChannelId" in comment else ""
            author_display_name = comment["authorDisplayName"]
            like_count = comment["likeCount"]
            published_at = comment["publishedAt"]
            updated_at = comment["updatedAt"]
            viewer_rating = comment["viewerRating"]
            is_reply_to = reply_to if reply_to is not None else ""
            writer.writerow(
                {'viewerRating': viewer_rating, 'videoId': video_id2, 'textDisplay': text_display,
                 'canReply': can_reply,
                 'totalReplyCount': total_reply_count, 'kind': kind, 'etag': etag, 'id': id2,
                 'authorChannelId': author_channel_id, 'authorDisplayName': author_display_name,
                 'likeCount': like_count,
                 'publishedAt': published_at, 'updatedAt': updated_at, 'isReplyTo': is_reply_to})

        try:
            results = youtube_service.commentThreads().list(
                part="snippet,replies",
                textFormat="plainText",
                maxResults=max_results,
                videoId=video_id
            ).execute()
            get_comment_page(results)
            pages = math.inf
            if isinstance(limit_pages, int) and limit_pages > 0:
                pages = limit_pages
            while 'nextPageToken' in results and pages > 0:
                pages -= 1
                # Fetch next page
                next_page_token = results['nextPageToken']
                try:
                    results = youtube_service.commentThreads().list(
                        part="snippet,replies",
                        textFormat="plainText",
                        maxResults=max_results,
                        pageToken=next_page_token,
                        videoId=video_id
                    ).execute()
                    # You have to ignore the first result
                    get_comment_page(results, ignore_first=True)
                except HttpError as err:
                    logger.error("HTTP Error for video_id,{0},pageToken,'{1}',The error was,{2}".format(video_id,
                                                                                                        next_page_token,
                                                                                                        str(err)))
                except Exception as err:
                    logger.error("Unexpected error:", sys.exc_info()[0], "Exception", err)
                    raise
        except HttpError as err:
            logger.error("HTTP Error for video_id,{0},firstPage,1,The error was,{1}".format(video_id,
                                                                                            str(err)))
        except socket.error as err:
            # We handle only Connection Reset by Peer currently
            if err.errno != errno.ECONNRESET:
                logger.error("Socket Error for video_id,{0},firstPage,1,The error was,{1}".format(video_id,
                                                                                                  str(err)))
                raise
            # Handle connection reset by peer by deleting last video downloaded and telling the user
            print("Connection reset by peer, stopped at video_id {0}".format(video_id))
            sys.exit(-1)

        except Exception as err:
            logger.error("Unexpected error:", sys.exc_info()[0], "Exception", err)
            raise

    return results


def get_default_youtube_service():
    """
    :return: default youtube api object, assumes there is a file ../data/devkey containing the developer api key
    """
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, "..", "data", "devkey"))
    if not os.path.isfile(filepath):
        raise FileNotFoundError("Developer key", filepath, "not found")

    developer_key = open(filepath).read()
    youtube_api_service_name = "youtube"
    youtube_api_version = "v3"

    youtube1 = build(youtube_api_service_name, youtube_api_version, developerKey=developer_key)
    return youtube1


if __name__ == "__main__":
    YOUTUBE = get_default_youtube_service()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--video_id_file', type=argparse.FileType('r'), help='File containing ids to download',
                        required=True)
    PARSER.add_argument('--csv_file', type=argparse.FileType('w'), help='CSV file', required=True)
    PARSER.add_argument('--start_from', type=int, default=1)
    PARSER.add_argument('--end', type=int, default=-1)
    PARSER.add_argument('--no-csv-header', action='store_false')
    ARGS = PARSER.parse_args()

    UNIQUE_VIDEOS = []
    START = (ARGS.start_from - 1) if ARGS.start_from > 0 else 0
    with open(ARGS.video_id_file.name) as f:
        for l in f:
            UNIQUE_VIDEOS += [l.strip()]
        END = ARGS.end if ARGS.end > START else len(UNIQUE_VIDEOS)
    SELECTED = UNIQUE_VIDEOS[START:END]

    print("Will download comments and replies from videos {0} to {1}".format((START + 1), END))
    CSV_FILE = str(ARGS.csv_file.name)
    PBAR = tqdm(range(END - START), desc="Downloading")

    for video in SELECTED:
        video_comment_threads = get_comment_threads(youtube_service=YOUTUBE, video_id=video,
                                                    csvfile=CSV_FILE, write_header=ARGS.no_csv_header)
        PBAR.update(1)
