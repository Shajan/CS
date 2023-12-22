#include <iostream>
#include "Animal.h"

int main() {
    Dog myDog;
    std::cout << "The animal is a " << myDog.name() << "." << std::endl;
    return 0;
}
