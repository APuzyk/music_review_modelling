from setuptools import setup, find_packages

setup(
    name='music_review_modeling',
    version='0.0.1',
    url='https://github.com/APuzyk/music_review_modelling',
    python_requires='>=3.5',
    install_requires=['keras', 'PyYAML', 'gensim', 'numpy', 'scikit-learn'],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'music_review_modeling=cli.__main__:main',
        ],
    }
)