from distutils.core import setup

setup(
    name='fire_behavior',
    version='0.1dev',
    description='Simulate 2-D wind driven fire in different conditions',
    author='Diane Wang',
    author_email='wangti68@msu.edu',
    packages=['fire_behavior',],
    license='MIT',
    #long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'matplotlib',
        'ipython',
        'pdoc3',
        'pip', 
        'pylint',
        'pytest',
        'scipy',
        'time'
    ])