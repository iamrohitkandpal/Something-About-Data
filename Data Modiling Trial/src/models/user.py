class User:
    
    def __init__(self, name, email, age):
        self.name = name
        self.email = email
        self.age = age
        
    def __str__(self):
        return f"User {self.name}, {self.email}, {self.age}"