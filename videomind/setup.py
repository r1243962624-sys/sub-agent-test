#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""VideoMind - 自动化视频内容处理系统"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="videomind",
    version="0.1.0",
    author="VideoMind Team",
    author_email="contact@videomind.ai",
    description="自动化视频内容处理系统 - 从视频链接到结构化笔记",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/videomind",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Text Processing :: Markup",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "videomind=videomind.cli.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "videomind": ["templates/*.json", "config/*.yaml"],
    },
)