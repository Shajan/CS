import java.util.*;

class Graph<T> {
  static private Random rand = new Random();

  public static void main(String args[]) {
    Graph g = createGraph(20);
    g.print();
  }

  static int randomVal() { return rand.nextInt(100); }

  static Graph<Integer> createGraph(int count) {
    Graph<Integer> g = new Graph<Integer>(new Node<Integer>(randomVal()));

    for (int i=1; i<count; ++i)
      addRandom(getRandomNode(g.root), new Node<Integer>(randomVal()));

    return g;
  }

  static void addRandom(Node<Integer> parent, Node<Integer> node) {
    int rem = rand.nextInt(4);

    switch (rem) {
      case 0: parent.addchild(node); break;
      case 1: addRandom(getRandomNode(parent), node); break;
      case 2: addRandom(getRandomNode(parent), node); break;
      case 3: addRandom(getRandomNode(parent), node); break;
      case 4: addRandom(getRandomNode(parent), node); break;
    }
  }

  static Node<Integer> getRandomNode(Node<Integer> node) {
    int c = node.childrenCount();
    if (c == 0)
      return node;
    else
      return node.child(rand.nextInt(c));
  }

  private Node<T> root;

  Graph(Node<T> n) { root = n; }

  void print() {
    StringBuilder sb = new StringBuilder();
    print(root, sb);
    System.out.println(sb.toString());
  }

  void print(Node<T> node, StringBuilder sb) {
    sb.append(String.format("(%s => [", node.data));
    String seperator = "";
    for (Node n : node.children) {
      sb.append(seperator);
      print(n, sb);
      seperator = ",";
    }
    sb.append("]");
  }

  static class Node<T> {
    T data;
    ArrayList<Node<T>> children;

    Node(T t) {
      data = t;
      children = new ArrayList<Node<T>>();
    }

    void addchild(Node n) { children.add(n); }
    Node<T> child(int i) { return children.get(i); }
    int childrenCount() { return children.size(); }

    String print() { return String.format("%s", data); }
  }
}
