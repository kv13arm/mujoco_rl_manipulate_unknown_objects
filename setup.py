from setuptools import setup

setup(
    name="Ant-Gripper-v0",
    version="0.0.1",
    install_requires=[
        "dm-control==1.0.3.post1",
        "gym==0.21.0",
        "scikit-learn==1.5.0",
        "opencv-python==4.8.1.78",
        "stable-baselines3[extra]",
        "torch==2.2.0",
        "scipy==1.8.1",
        "seaborn==0.11.2"
    ],
)
