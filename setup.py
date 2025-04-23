from setuptools import setup, find_packages

setup(
    name="Medical-Assistant",
    version="0.0.1",
    description="""
    About
    -----
    Medical Image Diagnosis Assistant with Visual Question Answering - the system responds intelligently based on the CNN's output + medical knowledge.
    """,
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/Medical-Assistant",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="Medical-Assistant, deep-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/Medical-Assistant/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/Medical-Assistant",
        "Source Code": "https://github.com/atikul-islam-sajib/Medical-Assistant",
    },
)
