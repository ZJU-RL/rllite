# coding: utf-8

from setuptools import find_packages, setup

setup(
    name='rllite',
    version='0.2.0',
    author='zjurl',
    author_email='wangyunkai.zju@gmail.com',
    url='https://github.com/ZJU-RL/rllite',
    description=u'rl lite repository',
    packages=find_packages(),
    install_requires=[
    	"utils",
    	"matplotlib",
    	"argparse",
    	"tqdm",
    	"typing",
    	"gym",
    	"scipy",
    	"tensorboardX"
    ]
)