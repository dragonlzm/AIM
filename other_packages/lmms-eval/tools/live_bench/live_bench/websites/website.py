import os
import time
from abc import ABC, abstractmethod

from webdriver_manager.core.driver import Driver

SUBJECT_MATCH = {
    "entertainment": "Entertainment",
    "artandculture": "Art & Culture",
    "entertainment": "Entertainment",
    "finance": "Economy & Finance",
    "politics": "Politics",
    "science": "Science",
    "sports": "Sports",
    "technology": "Technology",
}


class Website(ABC):
    def __init__(self, url=None, name=None, path=None, subject=None):
        self.url = url
        self.name = name
        self.path = path

        if subject in SUBJECT_MATCH:
            self.subject = SUBJECT_MATCH[subject]
        self.subject = subject

        assert self.url is not None or self.path is not None, "Either url or path must be provided"

    def get_path(self):
        if self.url:
            return self.url
        else:
            return self.path

    def visit(self, driver: Driver):
        self.pre_visit(driver)
        driver.get(self.url)
        self.post_visit(driver)

    def get_info(self):
        info = {}
        if self.url:
            info["url"] = self.url
        if self.name:
            info["name"] = self.name
        if self.subject:
            info["subject"] = self.subject
        return info

    @abstractmethod
    def pre_visit(self, driver: Driver):
        raise NotImplementedError("pre_action not implemented")

    @abstractmethod
    def post_visit(self, driver: Driver):
        raise NotImplementedError("post_action not implemented")


class DefaultWebsite(Website):
    def __init__(self, url, name=None):
        super().__init__(url, name)

    def pre_visit(self, driver: Driver):
        pass

    def post_visit(self, driver: Driver):
        time.sleep(5)  # Wait for 5 seconds to allow adblock to finish


class HumanScreenShotWebsite(Website):
    def __init__(self, name=None, path=None):
        try:
            image_name = os.path.basename(path)
            subject = image_name.split("_")[0]
        except:
            subject = None
        super().__init__(name=name, path=path, subject=subject)

    def pre_visit(self, driver: Driver):
        pass

    def post_visit(self, driver: Driver):
        pass
