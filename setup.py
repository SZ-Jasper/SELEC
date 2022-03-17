from setuptools import setup,find_packages
import sys, os

setup(name="selec",
      description="Selecting electrolyte for batteries",
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      version='1.0',
      author='Karen Li, Rose Lee, Bella Wu, Shuyan Zhao',
      url='https://github.com/SZ-Jasper/SELEC/tree/main/selec',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['numpy', 
                        'pandas', 
                        'scikit-learn', 
                        'streamlit', 
                        'plotly'],
      extras_require = {},
      packages=find_packages(),

      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
)