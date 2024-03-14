import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pwact", 
    version="0.0.5",
    author="LonxunQuantum",
    author_email="lonxun@pwmat.com",
    description="PWACT is an open-source automated active learning platform based on PWMLFF for efficient data sampling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LonxunQuantum/PWact",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'pwact = pwact.main:main'
        ]
    }
)

