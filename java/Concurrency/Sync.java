class Sync {
    public static void main(String args[]) {
      test_wait();
    }

    static void test_wait() {
        final Object o = new Object();

        Runnable a = new Runnable() {
            public void run() {
              System.out.println("a.run");
/*
              // Wake on notify
              try { Thread.sleep(100); } catch (InterruptedException e) { return; }
              System.out.println("a.notify.simple");
              synchronized(o) { o.notifyAll(); }

              try { Thread.sleep(1000); } catch (InterruptedException e) { return; }
              System.out.println("a.-----------");

              // Wake on timeout
              try { Thread.sleep(2000); } catch (InterruptedException e) { return; }
              System.out.println("a.notify.timeout (after 'b.wait.timeout.done')");
              synchronized(o) { o.notifyAll(); }
*/

              // Wake immediately
              synchronized(o) {
                System.out.println("a.lock.enter");
                System.out.println("a.notify.immediate");
                o.notifyAll(); 
                System.out.println("a.lock.exit");
              }
            }
        };

        Runnable b = new Runnable() {
            public void run() {
              System.out.println("b.run");
/*
              // Wake on notify
              try { synchronized(o) { o.wait(200); } } catch (InterruptedException e) { return; }
              System.out.println("b.wait.simple.done (after 'a.notify.simple')");

              try { Thread.sleep(1000); } catch (InterruptedException e) { return; }
              System.out.println("b.-----------");

              // Wake on timeout
              try { synchronized(o) { o.wait(100); } } catch (InterruptedException e) { return; }
              System.out.println("b.wait.timeout.done");
*/

              // Wake immediately
              try {
                synchronized(o) {
                  System.out.println("b.lock.enter");
                  o.wait(50000);
                  System.out.println("b.wait.immediate.done (after 'a.notify.immediate')");
                  System.out.println("b.lock.exit");
                }
              } catch (InterruptedException e) { return; }
            }
        };

        new Thread(a).start();
        new Thread(b).start();
    }
}
