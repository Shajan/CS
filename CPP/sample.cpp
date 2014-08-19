#include <stdio.h>

void unary_operator();
bool file_io(const char* file_name);

int main(int argc, char* argv[]) {
    unary_operator();
    file_io("test.data");
    return 0;
}

// Test Unary operator
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

bool file_io(const char* file_name) {
    FILE* f = fopen(file_name, "w");
    if (f == NULL) {
        printf("Unable to create file %s\n", file_name);
        return false;
    }
    if (fputs("Hello World!\nSecond Line\n", f) < 0) {
        printf("Unable to write to file %s\n", file_name);
        fclose(f);
        return false;
    }
    fclose(f);

    f = fopen(file_name, "r");
    char buffer[20];
    // Read the first 5 chars, appends a null char at the end
    if (fgets(buffer, 6, f) == NULL) {
        printf("Unable to read from file %s\n", file_name);
        fclose(f);
        return false;
    }
    printf("First 5 chars :%s\n", buffer);

    // Read the first line, including new line, appends null char at end
    if (fgets(buffer, 20, f) == NULL) {
        printf("Unable to read from file %s\n", file_name);
        fclose(f);
        return false;
    }
    printf("Rest of the first line :%s", buffer);
    fclose(f);
    return true;
}

