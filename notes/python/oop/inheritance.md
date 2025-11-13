---
title: Inheritance and Polymorphism
course: python
chapter: oop
order: 2
tags: [python, oop, inheritance, polymorphism]
status: not-started
readTime: 10 min read
---

# Inheritance and Polymorphism

Inheritance is one of the pillars of Object-Oriented Programming, allowing classes to inherit properties and methods from other classes.

## Basic Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks!"
```

## Multiple Inheritance

Python supports inheriting from multiple classes:

```python
class Flyer:
    def fly(self):
        return "Flying!"

class Swimmer:
    def swim(self):
        return "Swimming!"

class Duck(Flyer, Swimmer):
    pass

duck = Duck()
print(duck.fly())
print(duck.swim())
```

## Method Resolution Order (MRO)

Python uses C3 linearization for method resolution:

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

print(D.mro())  # Shows the method resolution order
```

## Abstract Base Classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2
```

## Polymorphism in Action

```python
def calculate_total_area(shapes):
    total = 0
    for shape in shapes:
        total += shape.area()
    return total

shapes = [Square(5), Circle(3), Rectangle(4, 6)]
print(calculate_total_area(shapes))
```

## Conclusion

Inheritance and polymorphism are powerful tools for code reuse and flexibility. Use them wisely to create maintainable and extensible code.
