import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(include=("pyminion", "pyminion.*"))

setuptools.setup(
    name="pyminion",
    version="0.5.0",
    author="Evan Slack",
    author_email="evan.slack@outlook.com",
    description="Dominion but make it python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evanofslack/pyminion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["pyminion_master", *(f"pyminion_master.{package}" for package in packages)],
    package_dir={"pyminion_master": "."},
    python_requires=">=3.10",
)
