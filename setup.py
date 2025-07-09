# setup.py
from pathlib import Path
from setuptools import setup, find_packages

def _read_requirements():
    lines = (Path(__file__).with_name("requirements.txt")
             .read_text(encoding="utf-8").splitlines())
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return lines

setup(
    name="coladan",
    version="0.1.0",
    author="Zijun Wang",
    author_email="wangzijun_kenny@163.com",
    description="Coladan multimodal gene-image model",
    packages=find_packages(),     
    install_requires=_read_requirements(),
    python_requires="==3.10.0",
)
