/**
 * Types
 *  bool, byte, i16, i32, i64, double
 *  string, binary
 *  map<type1,type2>, list<type>, set<type>
 *
 * Composition
 *   struct MyStruct {
 *     1: string name = "unknown",
 *     2: string value,
 *     3: optional i16 id = 0,
 *   }
 *
 * Service
 *   service MyService { 
 *     void doNothing(),
 *     string ping(1:string str)
 *   }
 *
 * Generate CPP and java files (in gen-cpp & gen-java folder)
 *   thrift -r --gen cpp java sample.thrift 
 */

namespace cpp serializer
namespace java serializer

# Reguar thrift compiler does not understand scala and will fail
# on the next line. scrooge thrift compiler will pick up '#@'
#@namespace scala serializer

struct KeyVal {
  1: string key,
  2: string val 
}

