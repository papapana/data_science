"""
Setup file for module
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Data Science Project to test default functionality',
    'author': 'Vassilis Papapanagiotou',
    'url': '',
    'download_url': '',
    'author_email': 'vassilis.papapanagiotou@nexiot.ch',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': [],
    'scripts': [],
    'name': 'yelp_ds'
}

setup(**config)