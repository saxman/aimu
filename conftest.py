import logging

def pytest_configure(config):
    logging.getLogger("aimu").setLevel(logging.DEBUG)