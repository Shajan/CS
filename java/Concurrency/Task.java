import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

class Task {
    static Random sRandom;
    static int sThreadCount; // For thread pool
    static int sIterations;
    static int sSleepMaxMs;
    static CountDownLatch sLatch;

    public static void main(String args[]) {
        sThreadCount = 4;
        sIterations = 1000; 
        sSleepMaxMs = 20; 
        sRandom = new Random(System.currentTimeMillis()); 
        sLatch = new CountDownLatch(sIterations);
        Task t = new Task();

        long duration = 0;
        duration = t.useThreads();
        System.out.println("Threads : " + duration/(1000*1000) + "ms");
        duration = t.useThreadPool();
        System.out.println("ThreadPool : " + duration/(1000*1000) + "ms");
        duration = t.useThreadPoolWrongWay();
        System.out.println("ThreadPoolWrong : " + duration/(1000*1000) + "ms");
    }

    public long useThreads() {
        long start = System.nanoTime();
        // Using one thread per task
        //System.out.println("mainThread::Creating runnable object");

        Runnable runnable = new Runnable() {
            public void run(){
                //System.out.println("newThread::Start");
                try {
                    Thread.sleep(sRandom.nextInt(sSleepMaxMs)); 
                } catch (Exception e) {
                    System.out.println("newThread::exception " + e);
                }
                sLatch.countDown();
                //System.out.println("newThread::End");
            }
        };

        //System.out.println("mainThread::Launching new thread");
        for(int i=0; i<sIterations; i++){
            new Thread(runnable).start();
        }
        //System.out.println("mainThread::Done");
        try {
            sLatch.await();
        } catch (Exception e) {
            System.out.println("mainThread::exception " + e);
        }
        return System.nanoTime() - start;
    }
 
    public long useThreadPool() {
        long start = System.nanoTime();
        // Using thread pool
        //System.out.println("newThreadPool::Start");
        ExecutorService threadPool = Executors.newFixedThreadPool(sThreadCount);
        CompletionService<String> taskCompletionService = new ExecutorCompletionService<String>(threadPool);

        for(int i=0; i<sIterations; i++){
           taskCompletionService.submit(new RandomSleepTask());
        }

        for(int i=0; i<sIterations; i++){
            try {
                // Blocking call take(), will wait for atleast one task to be done.
                String delay = taskCompletionService.take().get(); // <-- pick tasks that are done
                //System.out.println("mainThread::GotResult " + delay);
            } catch (Exception e) {
                System.out.println("mainThread::exception " + e);
            }
        }
        threadPool.shutdown();
        //System.out.println("newThreadPool::End");
        return System.nanoTime() - start;
    }

    public long useThreadPoolWrongWay() {
        long start = System.nanoTime();
        // Using thread pool
        //System.out.println("newThreadPool::Start");
        ExecutorService threadPool = Executors.newFixedThreadPool(sThreadCount);
        List<Future<String>> futures = new ArrayList<Future<String>>(sIterations);

        for(int i=0; i<sIterations; i++){
            futures.add(threadPool.submit(new RandomSleepTask()));
        }

        for (Future<String> future : futures) {
            try {
                // Blocking call get(), will wait for that particular to be done
                String delay = future.get(); // <-- short tasks will just be waiting behind long running ones
                //System.out.println("mainThread::GotResult " + delay);
            } catch (Exception e) {
                System.out.println("mainThread::exception " + e);
            }
        }
        threadPool.shutdown();
        //System.out.println("newThreadPool::End");
        return System.nanoTime() - start;
    }

    private final class RandomSleepTask implements Callable<String> {
        public String call(){
            long delay = sRandom.nextInt(sSleepMaxMs); 
            //System.out.println("Task::Start " + delay);
            try {
                Thread.sleep(delay);
            } catch (Exception e) {
                System.out.println("Task::exception " + e);
            }
            //System.out.println("Task::End " + delay);
            return String.valueOf(delay);
        }
    }
}

