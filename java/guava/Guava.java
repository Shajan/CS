import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

class Guava {
  public static void main(String args[]) {
    cache();
  }

  private static void cache() {
    rateLimitter();
  }

  private static void rateLimitter() {
    // One item per 100 ms, capacity 2
    RateLimitter<String> limitter = new RateLimitter(1, 100, 2);
    
    if (!limitter.check("a")) System.out.println("fail");
    if (limitter.check("a")) System.out.println("fail");
    if (!limitter.check("b")) System.out.println("fail");
    if (limitter.check("b")) System.out.println("fail");
    limitter.print(); // {b=2, a=2}

    // At this point, we are at max capaicty, "a" is the oldest and gets evicted
    if (!limitter.check("c")) System.out.println("fail");
    if (limitter.check("c")) System.out.println("fail");

    // Keeping 'b' alive
    if (limitter.check("b")) System.out.println("fail");

    if (!limitter.check("d")) System.out.println("fail");
    if (limitter.check("d")) System.out.println("fail");

    // Keeping 'b' alive
    if (limitter.check("b")) System.out.println("fail");

    if (!limitter.check("e")) System.out.println("fail");
    if (limitter.check("e")) System.out.println("fail");
    limitter.print(); // {e=2, b=4}

    // Wait for caches to timeout
    try { Thread.sleep(100); } catch (Exception e) { } 
    limitter.print(); // {}
  }
}

// Use Cache for rate limitting
// Resource can be a user, file etc.
class RateLimitter<Resource> {

  LoadingCache<Resource, AtomicInteger> limitter = null;
  int rate = 0;

  // Number of operations (rate), per timeWindowMs
  // Capacity is max number of resources per timewindow to monitor
  public RateLimitter(int rate, int timeWindowMs, int capacity) {
    this.rate = rate;
    this.limitter = CacheBuilder
      .newBuilder()
      .initialCapacity(capacity)
      .maximumSize(capacity) // avoid resize
      .expireAfterAccess(timeWindowMs, TimeUnit.MILLISECONDS)
      .build(
        new CacheLoader<Resource, AtomicInteger>() {
          public AtomicInteger load(Resource key) { return new AtomicInteger(0); } // Initial value
        }
      );
  }

  public boolean check(Resource key) {
    return (limitter.getUnchecked(key).incrementAndGet() <= rate);
  }

  public void print() {
    System.out.println(limitter.asMap());
  }
}
