#!/usr/bin/env python3

import argparse
import csv
import errno
import logging
import math
import os
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
    fh = logging.FileHandler('youtube_comment_errors.log')
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    results = None

    # Setup csv file
    fieldnames = ['videoId', 'textDisplay', 'isReplyTo', 'canReply', 'totalReplyCount', 'kind', 'etag', 'id',
                  'authorChannelId', 'authorDisplayName', 'likeCount', 'publishedAt', 'updatedAt', 'viewerRating']
    with open(csvfile, 'a') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        def get_comment_page(page_results, ignore_first=False):
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
            if type(limit_pages) is int and limit_pages > 0:
                pages = limit_pages
            while ('nextPageToken' in results) and (pages > 0):
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
                    pass
                except Exception as e:
                    logger.error("Unexpected error:", sys.exc_info()[0], "Exception", e)
                    raise
        except HttpError as err:
            logger.error("HTTP Error for video_id,{0},firstPage,1,The error was,{1}".format(video_id,
                                                                                                str(err)))
            pass
        except SocketError as e:
            # We handle only Connection Reset by Peer currently
            if e.errno != errno.ECONNRESET:
                logger.error("Socket Error for video_id,{0},firstPage,1,The error was,{1}".format(video_id,
                                                                                                  str(err)))
                raise
            # Handle connection reset by peer by deleting last video downloaded and telling the user
            print("Connection reset by peer, stopped at video_id {0}".format(video_id))
            sys.exit(-1)

        except Exception as e:
            logger.error("Unexpected error:", sys.exc_info()[0], "Exception", e)
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

    youtube = build(youtube_api_service_name, youtube_api_version, developerKey=developer_key)
    return youtube


if __name__ == "__main__":
    youtube = get_default_youtube_service()
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id_file', type=argparse.FileType('r'), help='File containing ids to download', required=True)
    parser.add_argument('--csv_file', type=argparse.FileType('w'), help='CSV file', required=True)
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--no-csv-header', action='store_false')
    args = parser.parse_args()

    unique_videos = []
    start = (args.start_from - 1) if args.start_from > 0 else 0
    with open(args.video_id_file.name) as f:
        for l in f:
            unique_videos += [l.strip()]
        end = args.end if args.end > start else len(unique_videos)
    selected = unique_videos[start:end]

    print("Will download comments and replies from videos {0} to {1}".format((start + 1), end))
    csv_file = str(args.csv_file.name)
    pbar = tqdm(range(end-start), desc="Downloading")

    for video in selected:
        video_comment_threads = get_comment_threads(youtube_service=youtube, video_id=video,
                                                    csvfile=csv_file, write_header=args.no_csv_header)
        pbar.update(1)



