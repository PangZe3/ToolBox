from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'ToolBox for Adversarial Deep Learning'
LONG_DESCRIPTION = 'Some handy code for adversarial deep learning'

# 配置
setup(
    name="ToolBox",
    version=VERSION,
    author="PangZe3",
    author_email="<2221511784@qq.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],

    keywords=['python', 'adversarial deep learning'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ]
)

