from setuptools import setup, find_packages

def load_requirements(path):
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="medibot-package",
    version="0.1.0",
    package_dir={"": "src"},         # tell setuptools to look in src/
    packages=find_packages("src"),   # search for packages inside src/
    install_requires=load_requirements("requirements.txt"),
    python_requires=">=3.8",
)
