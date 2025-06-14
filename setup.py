from setuptools import setup, find_packages

setup(
    name="image-search-app",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'app': ['templates/*', 'static/*', 'static/css/*', 'static/js/*', 'static/images/*']
    },
    include_package_data=True,
    install_requires=[
        'flask>=2.0.1',
        'flask-cors>=3.0.10',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'faiss-cpu>=1.7.0; sys_platform != "darwin"',
        'faiss-gpu>=1.7.0; sys_platform != "darwin"',  # For GPU support
        'pillow>=8.3.1',
        'numpy>=1.21.0',
        'open-clip-torch>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'image-search=app.__main__:main',
        ],
    },
    include_package_data=True,
    python_requires='>=3.7',
)
