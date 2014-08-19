#include <stdio.h>
#include <functional>

void std_function();
void lambda();

void function_ptr() {
    std_function();
    lambda();
}

// Functions that can be called
void println(const char* s) { printf("%s\n", s); }
int increment(int i) { return ++i; }

void std_function() {
    std::function<void(const char*)> f_log = println;
    f_log("std::function<void(const char*)>..");

    std::function<int(int)> f_increment = increment;
    printf("std::function<int(int)>.. increment(10)\n-->%d\n", f_increment(10));
}

void lambda() {
    // Define function, assign to variable 
    auto f1 = [] () { printf("Basic lambda\n"); };
    // Call the function
    f1();

    // Compiler figures out the retun type
    auto incr = [] (int i) { return ++i; };
    printf("incr = [] (int i) { return ++i; }; incr(10);\n-->%d", incr(10));
}
