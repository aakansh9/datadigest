from setuptools import setup

with open("requirements.txt") as f:
    req = f.read().splitlines()
    req = list(filter(lambda x: not x.startswith("#"), req))

setup(
    name="datadigest",
    version="0.0.1",
    packages=["datadigest"],
    package_dir={'': 'src'},
    install_requires=req,
)
