from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['mdsimulator']
    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    cmdclass={'test': PyTest},
    name='mdsimulator',
    version='0.1.0',
    author='Marcel Hinsche, Leon Klein, Zeno',
    author_email='marcel.hinsche@fu-berlin.de',
    url='https://github.com/Marsll/md-simulator',
    long_description=open('README.txt').read(),
    packages=['mdsimulator', 'mdsimulator.test'],
    setup_requires=['pytest-runner',],
    install_requires=['numpy'],
    tests_require=['pytest'],
    zip_safe=False)
