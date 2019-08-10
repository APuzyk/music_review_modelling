from setuptools import setup

setup(
    name='music_review_modeling',
    entry_points={
        'console_scripts': [
            'music_review_modeling=command_line:main',
        ],
    }
)