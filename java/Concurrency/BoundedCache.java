import java.util.concurrent.ConcurrentHashMap;
//import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ArrayBlockingQueue;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.Cache;

class BoundedCache {
  public static void main(String args[]) {
    test();
  }

  static void test() {
    int mapCapacity = 10;
    int arrayCapacity = 10;

    Cache<String, ArrayBlockingQueue<Integer>> cache = CacheBuilder.newBuilder()
      .initialCapacity(mapCapacity)
      .maximumSize(mapCapacity) // avoid resize
      .build();

    for (int i=0; i<mapCapacity; ++i) {
      cache.put(String.format("%d", i), new ArrayBlockingQueue<Integer>(arrayCapacity));
    }
    print(cache);

    cache.put(String.format("%d", mapCapacity), new ArrayBlockingQueue<Integer>(arrayCapacity));
    print(cache);

    ArrayBlockingQueue<Integer> items = cache.getIfPresent("1");
    for (int i=0; i<arrayCapacity; ++i) {
      items.add(i);
    }
    print(items);

    try {
      items.add(arrayCapacity);
    System.out.println("Unexpected!");
    } catch (IllegalStateException ex) {
    }
    print(items);
  }

  static void print(Cache<String, ArrayBlockingQueue<Integer>> cache) {
    System.out.println(String.format("Size: %d", cache.size()));
    for (String key : cache.asMap().keySet()) {
      System.out.println(key);
    }
  }

  static void print(ArrayBlockingQueue<Integer> items) {
      System.out.println(items.toString());
  }
}
