# test file for git 
from typing import List, Set, Optional, Callable

class TestClass(object):
    """Class to generate a gmail id for a person
    """    
    def __init__(self, first_name:str, last_name:str, middle_name:Optional[str]=None):
        """method to initialize `TestClass`

        Args:
            first_name (str): first name of the person
            last_name (str): last name of the person
            middle_name (Optional[str], optional): middle name of the person. Defaults to None.
        """        
        self.first_name = first_name
        self.last_name = last_name
        self.middle_name = middle_name

    def generate_mail(self):
        """method to generate mail id

        Returns:
            str: gmail id of person
        """        

        if self.middle_name:
            return f"{self.first_name.lower()}.{self.middle_name.lower()}.{self.last_name.lower()}@gmail.com"
        else:
            return f"{self.first_name.lower()}.{self.last_name.lower()}@gmail.com"


if __name__ == "__main__":
    
    FIRST = 'Prashant'
    LAST = 'Mishra'

    test = TestClass(first_name=FIRST, last_name=LAST)
    mail = test.generate_mail()
    print(f"Generated mail id is: {mail}")
        