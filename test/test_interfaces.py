import unittest
from unittest.mock import MagicMock
from mr_modeling.interfaces.sqllite_interface import MusicReviewInterface


class TestSqliteInterface(unittest.TestCase):
    def test_get_uuid(self):
        mri = MusicReviewInterface('sqlite3', 'test')
        rv = [('5da72bec-581c-43b9-b756-64fefc62b58e',)]
        mri.get_query_results = MagicMock(return_value=rv)
        expect = '5da72bec-581c-43b9-b756-64fefc62b58e'
        self.assertEqual(mri.get_latest_uuid(), expect)

    def test_pull_text(self):
        mri = MusicReviewInterface('sqlite3', 'test')
        rv = [(1, '1~2~3'), (2, '3~4~5')]
        mri.get_query_results = MagicMock(return_value=rv)
        expect = {1: [1, 2, 3],
                  2: [3, 4, 5]}

        self.assertEqual(mri.pull_music_review_text('test', False), expect)

    def test_pull_word_dict(self):
        mri = MusicReviewInterface('sqlite3', 'test')
        rv = [(123, 'lets'), (345, 'test')]
        mri.get_query_results = MagicMock(return_value=rv)
        expect = {'lets': 123, 'test': 345}
        self.assertEqual(mri.pull_word_dict('test'), expect)

    def test_pull_metadata(self):
        mri = MusicReviewInterface('sqlite3', 'test')
        rv = [(1,
               'artist1',
               'genre1',
               'label1',
               'title1',
               'url1',
               8.6,
               0,
               'author1',
               'a_type1',
               1,
               7,
               5,
               2010,
               2010),
              (2,
               'artist2',
               'genre2',
               'label2',
               'title2',
               'url2',
               5.0,
               1,
               'author2',
               'a_type2',
               2,
               8,
               6,
               2011,
               2011)
              ]
        mri.get_query_results = MagicMock(return_value=rv)
        expect = {1: {
            'artist': 'artist1',
            'genre': 'genre1',
            'music_label': 'label1',
            'title': 'title1',
            'url': 'url1',
            'score': 8.6,
            'best_new_music': 0,
            'author': 'author1',
            'author_type': 'a_type1',
            'pub_weekday': 1,
            'pub_day': 7,
            'pub_month': 5,
            'pub_year': 2010,
            'year': 2010
        }, 2: {
            'artist': 'artist2',
            'genre': 'genre2',
            'music_label': 'label2',
            'title': 'title2',
            'url': 'url2',
            'score': 5.0,
            'best_new_music': 1,
            'author': 'author2',
            'author_type': 'a_type2',
            'pub_weekday': 2,
            'pub_day': 8,
            'pub_month': 6,
            'pub_year': 2011,
            'year': 2011
        }}
        self.assertEqual(mri.pull_review_metadata('test', False),
                         expect)
