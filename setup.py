"""Install setup."""
import setuptools

setuptools.setup(
    name="biorep_etl",
    version="0.0.1",
    url="git@github.com:ScottSnapperLab/biorep-etl.git",

    author="Gus Dunn",
    author_email="w.gus.dunn@gmail.com",

    description="Data cleaning and munging for the biorepository etc.",
    # long_description=open('README.rst').read(),

    packages=setuptools.find_packages('src'),
    package_dir={"": "src"},


    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    entry_points={
    "console_scripts": [
        "biorep-etl = biorep_etl.cli.main:run",
        ]
    },
)
