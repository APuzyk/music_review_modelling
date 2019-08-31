from setuptools import setup

setup(
    name='music_review_modeling',
    version='0.0.1',
    url='https://github.com/APuzyk/music_review_modelling',
    python_requires='>=3.5',
    install_requires=['keras'],

    entry_points={
        'console_scripts': [
            'music_review_modeling=command_line:main',
        ],
    }
)