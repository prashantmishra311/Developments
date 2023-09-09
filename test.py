# test file for git 
from typing import List, Set, Optional, Callable

class TestClass(object):

    def __init__(self, first_name:str, last_name:str, middle_name:Optional[str]=None):

        self.first_name = first_name
        self.last_name = last_name
        self.middle_name = middle_name

    def generate_mail(self):

        if self.middle_name:
            return f"{self.first_name.lower()}.{self.middle_name.lower()}.{self.last_name}@gmail.com"
        else:
            return f"{self.first_name.lower()}.{self.last_name}@gmail.com"


if __name__ = "__main__":
    
    FIRST = 'Prashant'
    LAST = 'Mishra'

    test = TestClass(first_name=FIRST, last_name=LAST)
    mail = test.generate_mail()
        