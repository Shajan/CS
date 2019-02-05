import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;
import java.util.concurrent.atomic.AtomicInteger;
import com.google.common.cache.*;

/*
 * ======================================================
 * 4 core 3.1 GHz Intel Core i7
 * Java 1.8.0_111, MacOS version 10.13,6
 * ------------------------------------------------------
 * [%get]--> 1         10        50        90        99        
 * ------------------------------------------------------
 * NoLock    185       91        123       137       96        
 * Exclusive 142       200       138       142       322       
 * RWLock    1631      5029      9414      879       194       
 * Concur    166       179       133       152       108
 * ======================================================
 */
// All values are in nano seconds
class ConcurrentHashPerf {
  public static void main(String args[]) {
    System.out.println(header());
    System.out.println(measure("NoLock", new NoLockCache()));
    System.out.println(measure("Exclusive", new ExclusiveLockCache()));
    System.out.println(measure("RWLock", new RWLockCache()));
    System.out.println(measure("Concur", new ConcurrentHashMapCache()));

    if (Util.errorCount() != 0) {
      System.out.println(String.format("Error count : %d", Util.errorCount()));
    }
  }

  static String header() {
    StringBuilder sb = new StringBuilder(Util.str("[%get]-->"));
    for (int percent : Config.GET_PERCENTS) {
      sb.append(Util.str(percent));
    }

    return sb.toString();
  }

  static String measure(String name, ICache cache) {
    StringBuilder sb = new StringBuilder(Util.str(name));
    for (int getPercent : Config.GET_PERCENTS) {
      LongSummaryStatistics avg = new LongSummaryStatistics();
      if (Config.verbose) {
        System.out.println(String.format("%s%s%s%s [%s, get:%d]",
          Util.str("Avg"), Util.str("Min"), Util.str("Max"), Util.str("Count"), name, getPercent));
      }
      for (int i=0; i<Config.SAMPLE_COUNT; ++i) {
        Job job = new Job(new SetTask(cache), new GetTask(cache), getPercent, Config.ITERATIONS);
        job.call();
        avg.accept(Math.round(job.avg()));
        if (Config.verbose) {
          System.out.println(String.format("%s%s%s%s",
            Util.str(job.avg()), Util.str(job.min()), Util.str(job.max()), Util.str(job.count())));
         }
      }
      if (Config.verbose) {
        System.out.println(String.format("%s %s %s",
          Util.str(name), Util.str(avg.getAverage()), Util.str(getPercent)));
      }
      sb.append(Util.str(avg.getAverage()));
    }
    return sb.toString();
  }
}

class Config {
  static final boolean verbose = false;
  static final int SAMPLE_COUNT = 5;
  static final int ITERATIONS = 1024*1024; // Number of times to measure for each sample
  static final int CACHE_KEY_MAX = 1024;
  static final int[] GET_PERCENTS = {1, 10, 50, 90, 99};
  static final int THREAD_COUNT = Runtime.getRuntime().availableProcessors();
}

class Util {
  static final Random random = new Random(System.currentTimeMillis());;
  static AtomicInteger error = new AtomicInteger();

  static int key() { return random.nextInt(Config.CACHE_KEY_MAX); }
  static boolean shouldGet(int getPercent) { return (random.nextInt(100) < getPercent); }
  static void error() { error.getAndIncrement(); }
  static int errorCount() { return error.get(); }
  static int value(int key) { return key + 10; }
  static boolean verify(int key, int value) { return value == value(key); }

  static String str(String str) { return String.format("%-10s", str); }
  static String str(int i) { return str(Integer.toString(i)); }
  static String str(long l) { return str(Long.toString(l)); }
  static String str(double d) { return str(Math.round(d)); }
}

interface ICache {
  void set(int key, Integer value);
  Integer get(int key);
}

class NoLockCache implements ICache {
  private Map<Integer,Integer> map = new HashMap<Integer,Integer>();

  public void set(int key, Integer value) { this.map.put(key, value); }
  public Integer get(int key) { return this.map.get(key); }
}

class ExclusiveLockCache implements ICache {
  private Map<Integer,Integer> map = new HashMap<Integer,Integer>();
  private Object lock = new Object();

  public void set(int key, Integer value) { synchronized (lock) { this.map.put(key, value); } }
  public Integer get(int key) { synchronized (lock) { return this.map.get(key); } }
}

class RWLockCache implements ICache {
  private Map<Integer,Integer> map = new HashMap<Integer,Integer>();
  private ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
  private Lock rLock = rwLock.readLock();
  private Lock wLock = rwLock.writeLock();

  public void set(int key, Integer value) {
    wLock.lock();
    try {
      this.map.put(key, value);
    } finally {
      wLock.unlock();
    }
  }
  public Integer get(int key) {
    rLock.lock();
    try {
      return this.map.get(key);
    } finally {
      rLock.unlock();
    }
  }
}

class ConcurrentHashMapCache implements ICache {
  private Map<Integer,Integer> map = new ConcurrentHashMap<Integer,Integer>();
  private Object lock = new Object();

  public void set(int key, Integer value) { this.map.put(key, value); }
  public Integer get(int key) { return this.map.get(key); }
}

class SetTask implements Callable<Long> {
  ICache cache;

  SetTask(ICache cache) { this.cache = cache; }

  // Returns latency in nano seconds of each call
  public Long call() {
    int key = Util.key();
    int val = Util.value(key);
    long start = System.nanoTime();
    cache.set(key, val);
    return System.nanoTime() - start;
  }
}

class GetTask implements Callable<Long> {
  ICache cache;

  GetTask(ICache cache) { this.cache = cache; }

  // Returns latency in nano seconds of each call
  public Long call() {
    int key = Util.key();
    long start = System.nanoTime();
    Integer value = cache.get(key);
    long duration = System.nanoTime() - start;
    if ((value != null) && !Util.verify(key, value))
      Util.error();
    return duration;
  }
}

class Job implements Callable<Boolean> {
  Callable<Long> getTask, setTask;
  int getPercent;
  int iterations;

  LongSummaryStatistics stats = new LongSummaryStatistics();
  long totalTime = 0;   // Wall clock time to execute the whole thing

  Job(Callable<Long> setTask, Callable<Long> getTask, int getPercent, int iterations) {
    this.getTask = getTask;
    this.setTask = setTask;
    this.getPercent = getPercent;
    this.iterations = iterations;
  }

  public Boolean call() {
    ExecutorService pool = Executors.newFixedThreadPool(Config.THREAD_COUNT);
    CompletionService<Long> taskCompletionService = new ExecutorCompletionService<Long>(pool);

    long start = System.nanoTime();
    for (int i=0; i<iterations; i++) {
      taskCompletionService.submit(Util.shouldGet(getPercent) ? getTask : setTask);
    }

    for (int i=0; i<iterations; i++) {
      try {
        long delay = taskCompletionService.take().get(); // <-- pick tasks that are done
        stats.accept(delay);
      } catch (InterruptedException|ExecutionException ex) {
        Util.error();
        totalTime = System.nanoTime() - start;
        pool.shutdown();
        ex.printStackTrace();
        return false;
      }
    }

    totalTime = System.nanoTime() - start;
    pool.shutdown();
    return true;
  }

  double avg() { return stats.getAverage(); }
  long max() { return stats.getMax(); }
  long min() { return stats.getMin(); }
  long count() { return stats.getCount(); }
}
