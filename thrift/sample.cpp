#include <stdio.h>

#include <boost/shared_ptr.hpp>

#include <protocol/TBinaryProtocol.h>
#include <protocol/TDenseProtocol.h>
#include <protocol/TJSONProtocol.h>
#include <transport/TTransportUtils.h>
#include <transport/TFDTransport.h>

#include "gen-cpp/KeyValService.h"

#define FILE_NAME "data.bin"

void read_from(const char* file_name, KeyVal& kv);
void write_to(const char* file_name, KeyVal& kv);

int main(int argc, char* argv[]) {
    getchar();

    bool read=true;
    const char * file_name = FILE_NAME;
    KeyVal kv;

    if (argc > 2 && strcmp(argv[1], "write")) {
        read = false; 
    }

    if (read) {
      read_from(file_name, kv);
    } else {
      write_to(file_name, kv);
    }
    return 0;
}

void read_from(const char* file_name, KeyVal& kv) {
}

void write_to(const char* file_name, KeyVal& kv) {
}

