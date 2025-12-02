from setuptools import setup, find_packages

setup(
    name="markboard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "plotly",
        "ipywidgets",
        "anywidget"
    ],
    include_package_data=True,
)