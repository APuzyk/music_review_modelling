pip uninstall mr_modeling
rm -r ~/Projects/music_reviews/music_review_modeling/dist
python3 ~/Projects/music_reviews/music_review_modeling/setup.py sdist bdist_wheel
pip install ~/Projects/music_reviews/music_review_modeling/dist/mr_modeling-0.0.1-py3-none-any.whl
