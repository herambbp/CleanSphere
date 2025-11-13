---
title: Classes and Objects in Python
course: python
chapter: oop
order: 1
tags: [python, oop, classes, objects]
status: in-progress
readTime: 12 min read
---

# Classes and Objects in Python

Object-Oriented Programming (OOP) is a programming paradigm that organizes code into objects and classes. Python supports OOP principles elegantly and intuitively.

## What is a Class?

A **class** is a blueprint for creating objects. It defines attributes (data) and methods (functions) that the objects will have.

### Basic Class Syntax

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says Woof!"
```

## Creating Objects

An **object** is an instance of a class:

```python
# Creating objects
buddy = Dog("Buddy", 3)
max_dog = Dog("Max", 5)

# Using object methods
print(buddy.bark())  # Output: Buddy says Woof!
print(max_dog.bark())  # Output: Max says Woof!
```

## The `__init__` Method

The `__init__` method is a special method called a **constructor**. It's automatically called when you create a new object.

### Constructor Parameters

```python
class Person:
    def __init__(self, name, age, city):
        self.name = name  # Instance variable
        self.age = age
        self.city = city

    def introduce(self):
        return f"Hi, I'm {self.name}, {self.age} years old from {self.city}"

# Create a person
alice = Person("Alice", 30, "New York")
print(alice.introduce())
```

## Instance Variables vs Class Variables

### Instance Variables

Unique to each object:

```python
class Car:
    def __init__(self, brand, model):
        self.brand = brand  # Instance variable
        self.model = model  # Instance variable
```

### Class Variables

Shared by all instances:

```python
class Car:
    wheels = 4  # Class variable

    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

car1 = Car("Toyota", "Camry")
car2 = Car("Honda", "Civic")

print(car1.wheels)  # 4
print(car2.wheels)  # 4
print(Car.wheels)   # 4
```

## Methods in Classes

### Instance Methods

Most common type, works with instance data:

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return f"Deposited ${amount}. New balance: ${self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            return "Insufficient funds"
        self.balance -= amount
        return f"Withdrew ${amount}. New balance: ${self.balance}"
```

### Class Methods

Use the `@classmethod` decorator:

```python
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes', 'basil'])

    @classmethod
    def pepperoni(cls):
        return cls(['mozzarella', 'tomatoes', 'pepperoni'])

# Create pizzas using class methods
pizza1 = Pizza.margherita()
pizza2 = Pizza.pepperoni()
```

### Static Methods

Use the `@staticmethod` decorator:

```python
class MathOperations:
    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def multiply(x, y):
        return x * y

# Call without creating an instance
result = MathOperations.add(5, 3)  # 8
```

## Encapsulation

Encapsulation is the bundling of data and methods that work on that data within a single unit (class).

### Private Attributes

Python uses naming conventions for privacy:

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary  # Private attribute

    def get_salary(self):
        return self.__salary

    def set_salary(self, new_salary):
        if new_salary > 0:
            self.__salary = new_salary

emp = Employee("John", 50000)
# print(emp.__salary)  # AttributeError
print(emp.get_salary())  # 50000
```

## Inheritance

Inheritance allows a class to inherit attributes and methods from another class.

### Basic Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())  # Buddy says Woof!
print(cat.speak())  # Whiskers says Meow!
```

### The `super()` Function

```python
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

class Car(Vehicle):
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)
        self.doors = doors

my_car = Car("Toyota", "Camry", 4)
```

## Polymorphism

Polymorphism allows objects of different classes to be treated as objects of a common base class.

```python
class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

shapes = [Rectangle(5, 4), Circle(3)]
for shape in shapes:
    print(f"Area: {shape.area()}")
```

## Special Methods (Magic Methods)

Python classes can define special methods for operator overloading:

### Common Magic Methods

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)

v1 = Vector(2, 3)
v2 = Vector(1, 4)
v3 = v1 + v2  # Uses __add__
print(v3)     # Uses __str__
```

## Mathematical Applications

Classes are perfect for representing mathematical concepts:

### Complex Numbers

```python
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(
            self.real + other.real,
            self.imag + other.imag
        )

    def magnitude(self):
        return (self.real**2 + self.imag**2)**0.5
```

The magnitude formula: $|z| = \sqrt{a^2 + b^2}$ where $z = a + bi$

## Property Decorators

The `@property` decorator creates getter methods:

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

    @property
    def circumference(self):
        return 2 * 3.14159 * self._radius

circle = Circle(5)
print(circle.area)  # Accessed like an attribute
```

## Best Practices

1. **Use meaningful class names**: CamelCase for classes
2. **Keep classes focused**: Single Responsibility Principle
3. **Use inheritance wisely**: Favor composition over inheritance when appropriate
4. **Document your classes**: Use docstrings

```python
class Student:
    """
    A class representing a student.

    Attributes:
        name (str): The student's name
        grade (int): The student's grade level
    """

    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
```

## Real-World Example

```python
class Library:
    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def find_book(self, title):
        for book in self.books:
            if book.title == title:
                return book
        return None

class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_available = True

    def checkout(self):
        if self.is_available:
            self.is_available = False
            return True
        return False

# Usage
library = Library("City Library")
book1 = Book("Python Crash Course", "Eric Matthes", "1234567890")
library.add_book(book1)
```

## Conclusion

Classes and objects are fundamental to Python programming. They allow you to:
- Organize code logically
- Model real-world entities
- Reuse code through inheritance
- Encapsulate data and behavior

Mastering OOP concepts will make you a more effective Python programmer and prepare you for advanced topics like design patterns and frameworks.
