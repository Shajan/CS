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

  public static void insert(BST parent, int val) {
    if (val > parent.val) {
      if (parent.right == null)
	parent.right = new BST(val);
      else
        insert(parent.right, val);
    } else {
      if (parent.left == null)
	parent.left = new BST(val);
      else
        insert(parent.left, val);
    }
  }

  public static void remove(int val) {
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

  public static void main(String[] args) {
    int[] a = {100, 110, 120, 115, 90, 105, 107, 95, 110, 117, 85};
    for (int i : a)
      insert(i);
    /*
    System.out.println("removing 90");
    remove(90);
    System.out.println("removing 115");
    remove(115);
    System.out.println("removing 110");
    remove(110);
    */
    print();
  }

  // Ascii art
  private static char leftCorner = 9492; // Looks like 'L'
  private static char sidewaysT = 9500;  // Looks like '|-'
  private static char horizontal = 9472; // Looks like '-'
  private static char vertical = 9474;   // Looks like '|'
  //private static String corner = new String(new char[] {leftCorner, horizontal, horizontal}); // L--
  //private static String joint = new String(new char[] {sidewaysT, horizontal, horizontal}); // |--

  // Printing tree
  public static void print() {
    if (root != null)
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
