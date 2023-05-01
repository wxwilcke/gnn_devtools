#!/usr/bin/env python

from setuptools import setup


version = '0.1.0'

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='gnn_devtools',
    version=version,
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    url='https://gitlab.com/wxwilcke/gnn_devtools',
    download_url = 'https://gitlab.com/wxwilcke/gnn_devtools/-/archive/' + version + '/gnn_devtools-' + version + '.tar.gz',
    description='Toolkit to generate random graph data to facilitate the development of graph neural networks with',
    long_description = open('README.md').read(),
    long_description_content_type="text/markdown",
    license='GLP3',
    include_package_data=True,
    zip_safe=True,
    keywords = ["gnn", "graph neural network", "graphs", "knowledge graphs"],
    packages=['gnn_devtools'],
    package_dir={"":"src"},
    install_requires = ["numpy"],
)
