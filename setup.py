from setuptools import setup

setup(
    name="gripper-env-v0",
    version="0.0.1",
    install_requires=[
        "dm-control==1.0.3.post1",
        "gym==0.25.0",
        "scikit-learn==1.1.1",
        "opencv-python==4.6.0.66",
    ],
)
