// animal.h
#ifndef ANIMAL_H
#define ANIMAL_H

#include <string>

// Abstract base class
class Animal {
public:
    // Pure virtual function
    virtual std::string name() const = 0;

    // Virtual destructor
    virtual ~Animal();
};

// Derived class
class Dog : public Animal {
public:
    std::string name() const;
};

#endif // ANIMAL_H
