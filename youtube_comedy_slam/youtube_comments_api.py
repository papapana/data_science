#!/usr/bin/env python3

import httplib2
import os
import sys
import pprint
import csv

from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

# CLIENT_SECRETS_FILE = "client_secrets.json"
DEVELOPER_KEY = "AIzaSyAzV-ojzFgf3NSi9V8v9f_Ew2xQcRHrF84"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)




def get_comment_threads(youtube, video_id, csvfile):
    """
    Our purpose is to download comments and append them to a csv with the following format:
    videoId,textDisplay, canReply, totalReplyCount,etag,id,authorChannelId,authorDisplayName,likeCount,publishedAt,updatedAt,viewerRating

    """
    results = youtube.commentThreads().list(
        part="snippet,replies",
        textFormat="plainText",
        maxResults=100,
        videoId=video_id
    ).execute()


    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(results)

    write_header = not os.path.isfile(csvfile)

    ## Setup csv file
    fieldnames = ['videoId', 'textDisplay', 'canReply', 'totalReplyCount', 'kind', 'etag', 'id', 'authorChannelId',
                  'authorDisplayName', 'likeCount', 'publishedAt', 'updatedAt', 'viewerRating']
    with open(csvfile, 'a') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        def get_comment_page(results, ignoreFirst = False):
            first = True
            for item in results["items"]:
                if first:
                    first = False
                    continue
                snippet = item["snippet"]
                comment = snippet["topLevelComment"]["snippet"]
                videoId = comment["videoId"]
                textDisplay = comment["textDisplay"]
                canReply = snippet["canReply"]
                totalReplyCount = snippet["totalReplyCount"]
                kind = item["kind"]
                etag = snippet["topLevelComment"]["etag"]
                id = snippet["topLevelComment"]["id"]
                authorChannelId = comment["authorChannelId"] if "authorChannelId" in comment else ""
                authorDisplayName = comment["authorDisplayName"]
                likeCount = comment["likeCount"]
                publishedAt = comment["publishedAt"]
                updatedAt = comment["updatedAt"]
                viewerRating = comment["viewerRating"]
                writer.writerow(
                    {'viewerRating': viewerRating, 'videoId': videoId, 'textDisplay': textDisplay, 'canReply': canReply,
                     'totalReplyCount': totalReplyCount, 'kind': kind, 'etag': etag, 'id': id,
                     'authorChannelId': authorChannelId, 'authorDisplayName': authorDisplayName, 'likeCount': likeCount,
                     'publishedAt': publishedAt, 'updatedAt': updatedAt, 'viewerRating': viewerRating, 'videoId': videoId})

        get_comment_page(results)

        counter = 0
        while ('nextPageToken' in results) and (counter < 3):
            # Fetch next page
            nextPageToken = results['nextPageToken']
            results = youtube.commentThreads().list(
                part="snippet,replies",
                textFormat="plainText",
                maxResults=100,
                pageToken = nextPageToken,
                videoId=video_id
            ).execute()
            # You have to ignore the first result
            get_comment_page(results, ignoreFirst=True)
            counter += 1

    return results


def get_comments(youtube, parent_id):
    results = youtube.comments().list(
        part="snippet",
        parentId=parent_id,
        textFormat="plainText"
    ).execute()

    print(results)

    for item in results["items"]:
        author = item["snippet"]["authorDisplayName"]
        text = item["snippet"]["textDisplay"]
        print("Comment by %s: %s" % (author, text))

    return results["items"]


def comment_threads_list_all_threads_by_channel_id(service, part, all_threads_related_to_channel_id):
    results = service.commentThreads().list(
        allThreadsRelatedToChannelId=all_threads_related_to_channel_id,
        part=part
    ).execute()

    print_results(results)


# comment_threads_list_all_threads_by_channel_id(youtube, 'snippet,replies', '7zCIRPQ8qWc')

video_comment_threads = get_comment_threads(youtube=youtube, video_id="7zCIRPQ8qWc", csvfile="data/dataset_movies.csv")
# parent_id = video_comment_threads[0]["id"]
# video_comments = get_comments(youtube, parent_id)




# # Call the search.list method to retrieve results matching the specified query term
#
# search_response = youtube.search().list(
#     q = "ALS Ice Bucket Challenge",
#     type = "video",
#     part = "id, snippet",
#     maxResults = 5
# ).execute()
#
# videos = {}
#
# # Add each result to the appropriate list, and then display the lists of matching videos
# # Filter out channels, and playlists
#
# for search_result in search_response.get("items", []):
#     if search_result["id"]["kind"] == "youtube#video":
#         # videos.append("%s" % (search_result["id"]["videoId"]))
#         videos[search_result["id"]["videoId"]] = search_result["snippet"]["title"]
#
# print("Videos:\n", "\n".join(videos), "\n")
