from networksecurity.logging import logging
from networksecurity.exception import NetworkSecurityException
import sys

try:
    a = 1/0
except Exception as e:
    raise NetworkSecurityException(e, sys)
