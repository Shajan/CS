import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.util.concurrent.RateLimiter;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

class CustomRateLimiter {
  public static void main(String args[]) {
    cache();
  }

  private static void cache() {
    //simpleRateLimiter();
    //testRequestTracker();
    hotSpotRateLimiter();
  }

  private static void sleep(long ms) { try { Thread.sleep(ms); } catch (Exception e) { } }

  private static void testRequestTracker() {
    RequestTracker tracker = new RequestTracker(10.0, 0);
    if (!tracker.isAllowed()) System.out.println("fail");
    if (tracker.isAllowed()) System.out.println("fail");

    tracker = new RequestTracker(10.0, 1);
    if (!tracker.isAllowed()) System.out.println("fail");
    if (tracker.isAllowed()) System.out.println("fail");

    tracker = new RequestTracker(10.0, 2);
    if (!tracker.isAllowed()) System.out.println("fail");
    if (!tracker.isAllowed()) System.out.println("fail");
    if (tracker.isAllowed()) System.out.println("fail");
    sleep(100);
    if (!tracker.isAllowed()) System.out.println("fail");
    if (tracker.isAllowed()) System.out.println("fail");
    sleep(100);
    if (!tracker.isAllowed()) System.out.println("fail");
  }

  private static void simpleRateLimiter() {
    // One item per 100 ms, capacity 2
    SimpleRateLimiter<String> limiter = new SimpleRateLimiter(1, 100, 2);
    
    if (!limiter.check("a")) System.out.println("fail");
    if (limiter.check("a")) System.out.println("fail");
    if (!limiter.check("b")) System.out.println("fail");
    if (limiter.check("b")) System.out.println("fail");
    limiter.print(); // {b=2, a=2}

    // At this point, we are at max capaicty, "a" is the oldest and gets evicted
    if (!limiter.check("c")) System.out.println("fail");
    limiter.print(); // {c=1, b=2}
    if (limiter.check("c")) System.out.println("fail");

    // Keeping 'b' alive
    if (limiter.check("b")) System.out.println("fail");

    if (!limiter.check("d")) System.out.println("fail");
    if (limiter.check("d")) System.out.println("fail");

    // Keeping 'b' alive
    if (limiter.check("b")) System.out.println("fail");

    if (!limiter.check("e")) System.out.println("fail");
    if (limiter.check("e")) System.out.println("fail");
    limiter.print(); // {e=2, b=4}

    // Wait for caches to timeout
    sleep(100);
    limiter.print(); // {}
  }

  private static void hotSpotRateLimiter() {
    // 10 qps, 1 second history, 2 recently used items to track
    HotSpotRateLimiter<String> limiter = new HotSpotRateLimiter(10.0, 1000, 1);

    /* Test throttling behavior */
    for (int i=0; i<10; ++i)
      if (!limiter.check("a")) System.out.println("fail");
    if (limiter.check("a")) System.out.println("fail");
    sleep(1000/10);
    if (!limiter.check("a")) System.out.println("fail");
    if (limiter.check("a")) System.out.println("fail");
    sleep(1000/10);
    if (!limiter.check("a")) System.out.println("fail");

    /* Test LRU behavior */
    limiter = new HotSpotRateLimiter(1.0, 100, 3);

    // Add 3 entries
    if (!limiter.check("a")) System.out.println("fail");
    if (!limiter.check("b")) System.out.println("fail");
    if (!limiter.check("c")) System.out.println("fail");

    if (limiter.check("a")) System.out.println("fail");
    if (limiter.check("b")) System.out.println("fail");
    if (limiter.check("c")) System.out.println("fail");

    // Touch "a" to bring it on the top of LRU
    if (limiter.check("a")) System.out.println("fail");

    // Adding "d" evicts "b", which is oldest now
    if (!limiter.check("d")) System.out.println("fail");

    // Check to see if "a" retained old state
    if (limiter.check("a")) System.out.println("fail");

    // "b" was evicted, so this one is totally new
    if (!limiter.check("b")) System.out.println("fail");
  }
}

// Use Cache for rate limitting
// Resource can be a user, file etc.
class SimpleRateLimiter<Resource> {

  LoadingCache<Resource, AtomicInteger> limiter = null;
  int rate = 0;

  // Number of operations (rate), per timeWindowMs
  // Capacity is max number of resources per timewindow to monitor
  public SimpleRateLimiter(int rate, int timeWindowMs, int capacity) {
    this.rate = rate;
    this.limiter = CacheBuilder
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
    return (limiter.getUnchecked(key).incrementAndGet() <= rate);
  }

  public void print() {
    System.out.println(limiter.asMap());
  }
}

class RequestTracker {
  AtomicInteger count;     // Current requests
  long ignoreInitial;      // Once count goes above this start throttling
  RateLimiter nextAllowed; // Used to find next allowed request once throttled

  public RequestTracker(double qps, long ignoreInitial) {
    this.ignoreInitial = ignoreInitial;
    this.count = new AtomicInteger(0);
    this.nextAllowed = RateLimiter.create(qps);
  }

  public boolean isAllowed() {
    // RateLimiter does not allow for bursty behavior, it has a strict cutoff, hence use of ignoreInitial
    if (count.incrementAndGet() >= ignoreInitial)
      return nextAllowed.tryAcquire();
    return true;
  }
}

class HotSpotRateLimiter<Resource> {
  // See http://docs.guava-libraries.googlecode.com/git/javadoc/src-html/com/google/common/util/concurrent/RateLimiter.html
  LoadingCache<Resource, RequestTracker> limiter = null;

  // qps: Number of operations allowed per second
  // timeWindowMs : Sliding window of time to track usage
  // Capacity : Max number of resources to monitor
  public HotSpotRateLimiter(final double qps, final long timeWindowMs, int capacity) {
    // If no activity is seen in this time window, stop tracking the object
    final long maxRequestsInTimeWindow = Math.round(qps*timeWindowMs/1000.0);

    this.limiter = CacheBuilder
      .newBuilder()
      .initialCapacity(capacity)
      .maximumSize(capacity) // avoid resize
      .expireAfterAccess(timeWindowMs, TimeUnit.MILLISECONDS)
      .build(
        new CacheLoader<Resource, RequestTracker>() {
          public RequestTracker load(Resource key) {
            return new RequestTracker(qps, maxRequestsInTimeWindow);
          }
        }
      );
  }

  public boolean check(Resource key) {
    return limiter.getUnchecked(key).isAllowed();
  }

  public void print() {
    System.out.println(limiter.asMap());
  }
}
