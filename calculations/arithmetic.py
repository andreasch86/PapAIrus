class Arithmetic:
    """
    A class for performing arithmetic operations.
    
    Args:
        a (int): The first operand.
        b (int): The second operand.
    
    Attributes:
        a (int): The first operand.
        b (int): The second operand.
    
    Methods:
        add(): Returns the sum of a and b.
        sub(): Returns the difference of a and b.
        mul(): Returns the product of a and b.
        div(): Returns the quotient of a and b.
        mod(): Returns the modulo of a and b.
    """
    a: int
    b: int
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        """
        Adds two numbers.
        
        Args:
            self: The object instance.
            a: The first number.
            b: The second number.
        
        Returns:
            The sum of a and b.
        
        Raises:
            None
        """
        return self.a + self.b

    def sub(self):
        """
        Subtracts self.b from self.a.
        
        Args:
            self: The instance of the class.
        
        Returns:
            The difference between self.a and self.b.
        
        Raises:
            None
        """
        return self.a - self.b

    def mul(self):
        """
        Multiplies two numbers.
        
        Args:
            self: The object instance.
            a (float): The first number.
            b (float): The second number.
        
        Returns:
            float: The product of a and b.
        
        Raises:
            None
        """
        return self.a * self.b

    def div(self):
        """
        Divides two numbers.
        
        Args:
            self: The object instance.
            a: The first number.
            b: The second number.
        
        Returns:
            The result of the division.
        
        Raises:
            ZeroDivisionError: If the divisor is zero.
        """
        try:
            c = float(self.a / self.b)
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")

        return c

    def mod(self):
        """
        Calculates the modulo of two numbers.
        
        Args:
            self: The object instance.
            a: The first number.
            b: The second number.
        
        Returns:
            The modulo of a and b.
        
        Raises:
            ZeroDivisionError: If b is 0.
        """
        try:
            c = float(self.a % self.b)
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
        return c