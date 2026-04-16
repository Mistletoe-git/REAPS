#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="REAPS",
    version="0.0.1",
    description="Receptor-Aware Peptide Sequence Designer",
    author="Yongkang Qiu",
    author_email="qiuyongkang@stu.scu.edu.cn",
    url="https://github.com/Mistletoe-git/REAPS",
    packages=find_packages(),
    py_modules=["train", "test", "inference"],
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_REAPS = train:main",
            "test_REAPS = test:main",
            "inference_REAPS = inference:main",
        ]
    },
)
