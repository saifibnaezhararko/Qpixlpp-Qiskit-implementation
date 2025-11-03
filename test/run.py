# test_main.py or conftest.py
import pytest
import sys

if __name__ == '__main__':
    # Run all tests in the current directory and subdirectories
    sys.exit(pytest.main(['-v', '--tb=short']))