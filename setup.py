from setuptools import setup


setup(
    name="open_couplet",
    version="1.0.1",
    description="AI for Chinese couplet",
    license="MIT Licence",
    author="QL Neo",
    author_email="neoql2333@163.com",

    packages=["open_couplet", "open_couplet.models"],
    install_requires=["torch==1.3.1"]
)
