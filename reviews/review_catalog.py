from interfaces.sqllite_interface import music_review_interface


class ReviewCatalogue:

    def __init__(self):
        self.interface = music_review_interface()
        self.uuid = self.interface.get_latest_uuid()
        self.review_content = None
        self.word_dict = None
        self.review_metadata = None

    def preprocess_reviews(self):
        self.pull_review_data()

    def pull_review_data(self):
        self.pull_review_content()
        self.pull_word_dict()

    def pull_review_content(self):
        self.review_content = self.interface.pull_music_review_text(self.uuid)

    def pull_word_dict(self):
        self.word_dict = self.interface.pull_word_dict(self.uuid)

    def pull_review_metadata(self):
        self.review_metadata = self.interface.pull_review_metadata(self.uuid)
