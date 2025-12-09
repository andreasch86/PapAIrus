class Arithmetic:
    """
    Class to perform arithmetic operations on two numbers.
    
    Args:
        a: The first number.
        b: The second number.
    
    Attributes:
        a: The first number.
        b: The second number.
    
    Methods:
        add(): Returns the sum of the two numbers.
        sub(): Returns the difference of the two numbers.
        mul(): Returns the product of the two numbers.
        div(): Returns the quotient of the two numbers.
        mod(): Returns the remainder of the division of the two numbers.
    """
    a: int
    b: int
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        """
        Returns the sum of the two numbers.
        
        Args:
            self: The Arithmetic object.
        
        Returns:
            The sum of the two numbers.
        """
        return self.a + self.b

    def sub(self):
        """
        Returns the difference of the two numbers.
        
        Args:
            self: The Arithmetic object.
        
        Returns:
            The difference of the two numbers.
        """
        return self.a - self.b

    def mul(self):
        """
        Returns the product of the two numbers.
        
        Args:
            self: The Arithmetic object.
        
        Returns:
            The product of the two numbers.
        """
        return self.a * self.b

    def div(self):
        """
        Returns the quotient of the two numbers.
        
        Args:
            self: The Arithmetic object.
        
        Returns:
            The quotient of the two numbers.
        
        Raises:
            ZeroDivisionError: If the divisor is zero.
        """
        try:
            c = float(self.a / self.b)
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")

        return c

    def mod(self):
        try:
            m = float(self.a % self.b)
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        return m