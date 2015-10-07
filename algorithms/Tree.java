import java.util.LinkedList;
import java.util.Queue;

class Tree<T> {
  T val;
  Tree<T> left;
  Tree<T> right;

  public Tree(T val, Tree<T> left, Tree<T> right) {
    this.val = val;
    this.left = left;
    this.right = right;
  }

  public Tree(T val) {
    this(val, null, null);
  }

  public String toString() {
    return val.toString();
  }
  
  public static int size(Tree t) {
    if (t == null)
      return 0;

    return size(t.left) + size(t.right) + 1;
  }

  public static int depth(Tree t) {
    if (t == null)
      return 0;

    return Math.max(depth(t.left), depth(t.right)) + 1;
  }

  public static void dfs(Tree t) {
    if (t == null)
      return;
    dfs(t.left);
    dfs(t.right);
    System.out.print(t + " ");
  }

  public static void bfs(Tree t) {
    if (t == null)
      return;
    Queue<Tree> q = new LinkedList<Tree>();
    q.add(t);
    while (!q.isEmpty()) {
      Tree node = q.remove();
      System.out.print(node + " ");
      if (node.left != null)
        q.add(node.left);
      if (node.right != null)
        q.add(node.right);
    }
  }

  public static void print(Tree t) {
    print("", t, true);
  }

  private static void print(String prefix, Tree t, boolean isLastSibling) {
    System.out.println(prefix + (isLastSibling ? "*---" : "+---") + "[" + t.toString() + "]");

    String paddNext = prefix + (isLastSibling ? "    " : "|   ");
    String paddCurrent = paddNext + "|"; 

    if (t.left != null) {
      System.out.println(paddCurrent);
      print(paddNext, t.left, t.right == null);
    }
    if (t.right != null) {
      System.out.println(paddCurrent);
      print(paddNext, t.right, true);
    }
  }

  public static void main(String[] args) {
    Tree t = new Tree<Integer>(10);

    t.left = new Tree<Integer>(5);
    t.left.left = new Tree<Integer>(2);
    t.left.left.right = new Tree<Integer>(4);
    t.left.right = new Tree<Integer>(7);

    t.right = new Tree<Integer>(15);
    t.right.left = new Tree<Integer>(12);
    t.right.left.right = new Tree<Integer>(14);
    t.right.right = new Tree<Integer>(17);

    System.out.println("........Depth........");
    System.out.print(depth(t));
    System.out.println("\n........DFS........");
    dfs(t);
    System.out.println("\n........BFS........");
    bfs(t);
    System.out.println("\n........Print........");
    print(t);
    System.out.println("\n........End........");

  }
}

