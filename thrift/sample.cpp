/*
 * Sample to demonstrate serialization / deserialization using thrift
 */
#include <stdio.h>
#include <iostream>

// Thrift requires boost
#include <boost/shared_ptr.hpp>

// Thrift libraries
#include <protocol/TJSONProtocol.h>
#include <transport/TFileTransport.h>

// Thrift generated code
#include "gen-cpp/KeyValService.h"

// File to read/write serilaized data
#define FILE_NAME "data.bin"

using namespace boost;
using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

void read_from(const char* file_name, KeyVal& kv);
void write_to(const char* file_name, KeyVal& kv);

int main(int argc, char* argv[]) {
    bool read=true;
    const char * file_name = FILE_NAME;
    KeyVal kv;

    if (argc > 1 && (strcmp(argv[1], "write") == 0)) {
        read = false; 
    }

    if (read) {
      read_from(file_name, kv);
      printf("key(%s), val(%s)", kv.key.c_str(), kv.val.c_str());
    } else {
      kv.key = "Name";
      kv.val = "Shajan Dasan";
      write_to(file_name, kv);
    }
    return 0;
}

void read_from(const char* file_name, KeyVal& kv) {
    shared_ptr<TFileTransport> transport(new TFileTransport(file_name, true));
    TJSONProtocol protocol(transport);
    kv.read(&protocol);
}

void write_to(const char* file_name, KeyVal& kv) {
    shared_ptr<TFileTransport> transport(new TFileTransport(file_name, false));
    TJSONProtocol protocol(transport);
    kv.write(&protocol);
}

