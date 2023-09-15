#!/usr/bin/env python

from setuptools import setup

setup(
    name="turntaking",
    version="0.0.0",
    description="Multimodal Voice Activity Prediction to turntaking",
    author="kazuyo Onishi",
    author_email="onishi.kazuyo.oi5@is.naist.jp",
    url="https://github.com/ahclab/turntaking",
    packages=[
        "turntaking",
        "turntaking.models",
        "turntaking.evaluation",
        "turntaking.dataload",
        "turntaking.vap_to_turn_taking",
    ],
)
