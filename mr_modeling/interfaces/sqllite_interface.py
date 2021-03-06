import sqlite3


class MusicReviewInterface:

    def __init__(self, c_type, loc):
        self.type = c_type
        self.loc = loc

    def get_query_results(self, query):
        conn = sqlite3.connect(self.loc)
        c = conn.cursor()
        c.execute(query)
        o = c.fetchall()
        c.close()
        return o

    def get_latest_uuid(self):
        query = """
                SELECT uuid
                FROM runs r
                JOIN (select max(end_ts) ets
                        from runs
                        where end_ts is not null
                            and run_type = 'review_prep')f
                    ON f.ets = r.end_ts
                    """
        return self.get_query_results(query)[0][0]

    def pull_music_review_text(self, uuid, is_test):
        query = """
            SELECT
                reviewid,
                contentVec
            FROM prepared_reviews
            WHERE uuid = '%s'
        """ % uuid
        query = query + ' limit 500' if is_test else query
        results = self.get_query_results(query)
        o = {}
        for row in results:
            o[row[0]] = [int(word_idx) for word_idx in row[1].split('~')]
        return o

    def pull_word_dict(self, uuid):
        query = """
            SELECT
                word_index,
                word
            FROM word_dictionary
            WHERE uuid = '%s'
        """ % uuid
        results = self.get_query_results(query)
        o = {}
        for ind, word in results:
            o[word] = ind

        return o

    def pull_review_metadata(self, uuid, is_test):
        columns = ['reviewid',
                   'artist',
                   'genre',
                   'music_label',
                   'title',
                   'url',
                   'score',
                   'best_new_music',
                   'author',
                   'author_type',
                   'pub_weekday',
                   'pub_day',
                   'pub_month',
                   'pub_year',
                   'year']
        query = """
                    SELECT
                        {columns}
                    FROM
                        prepared_reviews
                    WHERE
                        uuid = '{uuid}'
                    """.format(columns=','.join(columns),
                               uuid=uuid)
        query = query + ' limit 500' if is_test else query
        results = self.get_query_results(query)
        results = [list(i) for i in results]
        o = {}
        for i in results:
            to_add = {}
            for j in range(1, len(columns)):
                to_add[columns[j]] = i[j]
            o[i[0]] = to_add

        return o
