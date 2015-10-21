/*
 * Time Travelling HashMap
 * foo, bar  10
 * foo, bar2 100
 * foo, bar3 1000
 *    
 * foo 10  -> bar
 * foo 100 -> bar2
 * foo 500 -> bar2
 *  
 * 10/19/15
 *  1. Refactor for better unit testing
 *  2. mid = (start + end)/2
 *  3. binfind return index in arraylist, caller to seek back
 */

import java.util.*;

class VersionedMap {
  public static void main(String[] args) {
    TimeSeriesMap tMap = new TimeSeriesMap();
      
    tMap.set("a", "1", 10);
    tMap.set("b", "2", 100);
    tMap.print();

    assert tMap.get("a", 1) == null;
    assert tMap.get("a", 10) == "1";
    assert tMap.get("a", 100) == "1";
     
    assert tMap.get("b", 1) == null;
    assert tMap.get("b", 100) == "2";
   
    assert tMap.get("c", 100) == null;
    
    tMap.test();
  }
}

class TimeSeriesMap {
  private HashMap<String, Records> map = new HashMap<String, Records>();
       
  public void print() {
    System.out.println(map);
  }
          
  public void test() {
    new Records().test();
  }
      
  public String get(String key, int ts) {
    Records rs = map.get(key);    
    if (rs != null) {
      Record r = rs.get(ts);
      //System.out.println("..." + r.value);
      if (r != null) {
        return r.value;
      }
    }
    return null;
  }
                  
  public void set(String key, String value, int ts) {
    Records rs = null;
    if (map.containsKey(key)) {
      rs = map.get(key);
    } else {
      rs = new Records();
      map.put(key, rs);
    }
    rs.set(value, ts);
  }
}

class Record {
  String value;
  int timestamp;
  public Record(String value, int timestamp) {
    this.value = value;
    this.timestamp = timestamp;
  }
}

class Records {
  ArrayList<Record> al = new ArrayList<Record>();
  
  public Record get(int ts) {
    Record r = null;
    // TODO: Special Binary Search
    for (int i=0; i<al.size(); ++i) {
      if (ts >= al.get(i).timestamp)
      return al.get(i);
    }
    // Will we ever get here?
    //System.out.println("Getting " + r);
    return r;
  }
        
  public void set(String value, int ts) {
    Record rec = get(ts);
    if (rec != null && rec.timestamp == ts) {
      rec.value = value;
    } else {
      // TODO : assert order of ts
      //System.out.println("Setting " + value);
      al.add(new Record(value, ts));
    }
  } 
            
  public void test() {
    set("x", 10);
    set("y", 100);
    set("z", 1000);
    assert binFind(10) == 10;
    assert binFind(15) == 10;
    assert binFind(100) == 100;
  }
              
  private int binFind(int ts) {
    int start = 0;
    int end = 0;
    int mid = 0; 
    while (end > start) {
      mid = start + (end - start)/2;
      Record r = al.get(mid);
      if (ts > r.timestamp) {
        start = mid + 1;
      } else if (ts < r.timestamp) {
        end = mid - 1;
      } else {
        return r.timestamp;
      }
    }
    return al.get(start).timestamp;
  }
}
