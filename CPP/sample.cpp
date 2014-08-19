/*
 * Sample to demonstrate serialization / deserialization using thrift
 */
#include <stdio.h>
//#include <iostream>
//#include <fstream>

// Thrift requires boost
#include <boost/shared_ptr.hpp>

// Thrift libraries
#include <protocol/TJSONProtocol.h>
#include <transport/TFileTransport.h>

// Thrift generated code
#include "gen-cpp/sample_types.h"

// File to read/write serilaized data
#define FILE_NAME "data.bin"

using namespace boost;
using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

// To consume thrift generated code
using namespace serializer;

typedef unsigned char byte;

bool read_from(const char* file_name, KeyVal& kv);
bool read(const char* file_name, byte** ppb, int* pcb);
bool write_to(const char* file_name, KeyVal& kv);
bool write(const char* file_name, byte* pb, int cb);

int main(int argc, char* argv[]) {
    bool read=true;
    const char * file_name = FILE_NAME;
    KeyVal kv;

    if (argc > 1 && (strcmp(argv[1], "write") == 0)) {
        read = false; 
    }

    if (read) {
      if (!read_from(file_name, kv))
          return 1;
      printf("key(%s), val(%s)\n", kv.key.c_str(), kv.val.c_str());
    } else {
      kv.key = "Name";
      kv.val = "Shajan Dasan";
      if (!write_to(file_name, kv))
          return 1;
    }
    return 0;
}

bool read_from(const char* file_name, KeyVal& kv) {
    // Read from file
    byte* pb=NULL;
    int cb=0;
    if (!read(file_name, &pb, &cb))
        return false;

    // Deserialize
    shared_ptr<TMemoryBuffer> transport(new TMemoryBuffer(pb, cb));
    TJSONProtocol protocol(transport);
    kv.read(&protocol);

    return true;
}

bool write_to(const char* file_name, KeyVal& kv) {
    // Serialize
    shared_ptr<TMemoryBuffer> transport(new TMemoryBuffer());
    TJSONProtocol protocol(transport);
    kv.write(&protocol);

    // Write to file
    byte* pb=NULL;
    unsigned int cb=0;
    transport->getBuffer(&pb, &cb);
    return write(file_name, pb, cb);
}

bool read(const char* file_name, byte** ppb, int* pcb) {
    // FileTrasport appears to not interoperate with java code
    //shared_ptr<TFileTransport> transport(new TFileTransport(file_name, true));

    FILE* in = NULL;
    
    if ((in = fopen(file_name, "rb")) == NULL) {
        printf("Error unable to open file %s\n", file_name);
        return false;
    }

    fseek(in, 0, SEEK_END); 
    int cb = ftell(in);
    fseek(in, 0, SEEK_SET); 

    if (cb == 0) {
        printf("Error empty file %s\n", file_name);
        fclose(in);
        return false;
    }

    *ppb = new byte[cb];
    *pcb = fread(*ppb, sizeof(byte), cb, in); 
    fclose(in);

    if (*pcb != cb) {
        printf("Error reading file %s, bytes read:%d ,expected:%d\n", file_name, *pcb, cb);
        delete [] *ppb;
        *ppb = NULL;
        *pcb = 0;
        return false;
    }

    return true;
}

bool write(const char* file_name, byte* pb, int cb) {
    // FileTrasport appears to not interoperate with java code
    //shared_ptr<TFileTransport> transport(new TFileTransport(file_name, false));

    FILE* out = NULL;
    
    if ((out = fopen(file_name, "wb")) == NULL) {
        printf("Error unable to open file %s\n", file_name);
        return false;
    }

    int bytes = fwrite(pb, sizeof(byte), cb, out); 
    fclose(out);

    if (bytes != cb) {
        printf("Error writing file %s, bytes written:%d ,expected:%d\n", file_name, bytes, cb);
        return false;
    }

    fclose(out);
    return true;
}

