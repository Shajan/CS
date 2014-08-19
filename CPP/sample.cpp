#include <functional>

void function_ptr();
void unary_operator();
bool file_io(const char* file_name);

int main(int argc, char* argv[]) {
    function_ptr();
    unary_operator();
    file_io("test.data");
    return 0;
}

