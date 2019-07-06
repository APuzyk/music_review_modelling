import sqlite3


class MusicReviewInterface:

    def __init__(self, c_type='sqlite3', loc='/home/apuzyk/Projects/music_reviews/data/database.sqlite'):
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

    def pull_music_review_text(self, uuid):
        query = "SELECT reviewid, contentVec FROM prepared_reviews WHERE uuid = '%s'" % uuid
        l = self.get_query_results(query)
        o = {}
        for i in l:
            o[i[0]] = [int(j) for j in i[1].split('~')]
        return o

    def pull_word_dict(self, uuid):
        query = "SELECT word_index, word FROM word_dictionary WHERE uuid = '%s'" % uuid
        l = self.get_query_results(query)
        o = {}
        for ind, word in l:
            o[word] = ind

        return o

    def pull_review_metadata(self, uuid):
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
        l = self.get_query_results(query)
        l = [list(i) for i in l]
        o = {}
        for i in l:
            to_add = {}
            for j in range(len(columns)-1):
                to_add[columns[j+1]] = i[j+1]
            o[i[0]] = to_add

        return o
