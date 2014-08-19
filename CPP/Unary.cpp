#include <stdio.h>

// Unary operator
class A {
    friend void operator >> (int i, A& obj) {
        printf("%d >> A\n", i);
    }
    friend void operator << (int i, A& obj) {
        printf("%d << A\n", i);
    }
};

void unary_operator() {
    A a;
    10 >> a;
    25 << a;
}

