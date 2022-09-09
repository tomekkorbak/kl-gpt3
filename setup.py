from setuptools import setup
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="kl_gpt3",
    version="0.1.0",
    packages=["kl_gpt3"],
    license="LICENSE",
    description="",
    long_description=open("README.md").read(),
    install_requires=required
)