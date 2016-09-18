from unittest import TestCase
from testfixtures import LogCapture
import logging
from youtube_comments_api.youtube_comments_api import get_comment_threads
from youtube_comments_api.youtube_comments_api import get_default_youtube_service
import os


class TestGet_comment_threads(TestCase):
    def setUp(self):
        if os.path.isfile("data/test.csv"):
            os.remove('data/test.csv')

    def test_get_comment_threads(self):
        """
        Try downloading comments from a video that does not exist
        and one that does exist
        :return:
        """
        youtube = get_default_youtube_service()
        # First check for a video that does not exist
        with LogCapture(level=logging.ERROR) as l:
            results = get_comment_threads(youtube_service=youtube, video_id="oyP16Qg3UP4", csvfile="data/test.csv")
            assert results is None
            mystr = str(l)
            assert 'HTTP Error for video_id,oyP16Qg3UP4' in mystr

        # Now with a video that exists
        results = get_comment_threads(youtube_service=youtube, video_id="sNabaB-eb3Y", csvfile="data/test.csv",
                                      max_results=1)
        mystr = str(results)
        assert 'topLevelComment' in mystr

        # Now check if the replies are parsed
        results = get_comment_threads(youtube_service=youtube, video_id="-WVvOU1r2sM", csvfile="data/test.csv",
                                      max_results=10, limit_pages=1)
        mystr = open('data/test.csv').read()
        assert 'z12xjrcijsr4jnqhp04chxcptlbtjrn5zz00k.1473936165253594' in mystr


if __name__ == '__main__':
    unittest.main()