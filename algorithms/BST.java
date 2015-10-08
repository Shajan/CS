class BST {
  int val;
  BST left;
  BST right;

  static BST root;

  public BST(int val) {
    this.val = val;
  }

  public static void insert(int val) {
    if (root == null)
      root = new BST(val);
    else
      insert(root, val);
  }

  public static void insert(BST node, int val) {
    if (val > node.val) {
      if (node.right == null)
        node.right = new BST(val);
      else
        insert(node.right, val);
    } else {
      if (node.left == null)
        node.left = new BST(val);
      else
        insert(node.left, val);
    }
  }

  public static void remove(int val) {
    root = remove(root, val);
  }
  
  public static BST remove(BST node, int val) {
    if (node == null)
      return null;
    if (val > node.val)
      node.right = remove(node.right, val);
    else if (val < node.val)
      node.left = remove(node.left, val);
    else {
      if (node.left == null && node.right == null)
        return null;
      else if (node.left == null) 
        return node.right;
      else if (node.right == null)
        return node.left;
      else {
        // Both children are present.
        // Copy value of a child with one or zero children to node, then remove that child.
        // NOTE:
        //   Lowset value child on the right subtree will have atmost one child.
        //   Highest value child on the left will have at most one child.
        // Replace value of this node with one of the above two, then remove that node.

        // Find the highest value left child
	BST tmp = node.left;
	while (tmp.right != null)
	  tmp = tmp.right;
	node.val = tmp.val;
	node.left = remove(node.left, tmp.val); // ok to remove any child with that value
      }
    }
    return node;
  }

  public static BST find(int val) {
    BST current = root;

    while (current != null) {
      if (current.val == val)
        break;
      else if (val > current.val)
        current = current.right;
      else
        current = current.left;
    }

    return current;
  }

  private static void visit_ascending(BST t) {
    if (t == null)
      return;
    visit_ascending(t.left);
    System.out.print(t.val + ",");
    visit_ascending(t.right);
  }

  private static int visit_top(BST t, int n) {
    if (t == null || n == 0)
      return 0;
    int c = visit_top(t.right, n);
    if (c < n) {
      System.out.print(t.val + ",");
      ++c;
      c += visit_top(t.left, n - c);
    }
    return c;
  }

  private static int find_top(BST t, int n) {
    if (t == null || n == 0)
      return 0;
    int c = find_top(t.right, n);
    if (c < n) {
      if (n - c == 1)
        System.out.print(t.val + ",");
      ++c;
      c += find_top(t.left, n - c);
    }
    return c;
  }

  public static void main(String[] args) {
    //test_remove();
    //test_sort();
    test_top();
  }

  private static void setup_test_tree() {
    root = null;
    int[] a = {100, 110, 120, 115, 75, 111, 90, 105, 107, 95, 108, 117, 85};
    for (int i : a)
      insert(i);
  }

  private static void test_top() {
    setup_test_tree();
    System.out.println("top 10");
    visit_top(root, 10);
    System.out.println("\nfind 5th");
    find_top(root, 5);
    System.out.println("");
  }

  private static void test_sort() {
    setup_test_tree();
    print();
    visit_ascending(root);
    System.out.println("");
  }

  private static void test_remove() {
    setup_test_tree();
    print();
    System.out.println("removing 90");
    remove(90);
    print();
    System.out.println("removing 115");
    remove(115);
    print();
    System.out.println("removing 110");
    remove(110);
    print();
    remove(100);
    print();
  }

  // Ascii art
  private static char leftCorner = 9492; // Looks like 'L'
  private static char sidewaysT = 9500;  // Looks like '|-'
  private static char horizontal = 9472; // Looks like '-'
  private static char vertical = 9474;   // Looks like '|'

  // Printing tree
  public static void print() {
    print(root);
  }

  public static void print(BST t) {
    if (t != null)
      print("", horizontal, root, true);
  }

  private static void print(String prefix, char lr, BST t, boolean isLastSibling) {
    System.out.println(
      prefix + 
      (isLastSibling ?
        new String(new char[] {leftCorner, horizontal, lr, horizontal}) : 
        new String(new char[] {sidewaysT, horizontal, lr, horizontal})) + 
      "[" + t.val + "]");

    String paddNext = prefix + (isLastSibling ? "    " : vertical + "   ");
    String paddCurrent = paddNext + vertical; 

    if (t.left != null) {
      System.out.println(paddCurrent);
      print(paddNext, 'l', t.left, t.right == null);
    }
    if (t.right != null) {
      System.out.println(paddCurrent);
      print(paddNext, 'r', t.right, true);
    }
  }
}
