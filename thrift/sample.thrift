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
 * Generate CPP files (in gen-cpp folder)
 *   thrift --gen cpp sample.thrift 
 */

struct KeyVal {
  1: string key,
  2: string val 
}

service KeyValService {
  string getVal(1:KeyVal keyVal)
}

