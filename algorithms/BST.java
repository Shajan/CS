class BST {
  int val;
  BST left;
  BST right;
  BST parent;

  static BST root;

  public BST(BST parent, int val) {
    this.val = val;
    this.parent = parent;
  }

  public String toString() {
    return
    "value: " + val +
    ", left : " + ((left != null) ? left.val : "null") +
    ", right : " + ((right != null) ? right.val : "null");
  }

  public static void insert(int val) {
    if (root == null)
      root = new BST(null, val);
    else
      insert(root, val);
  }

  public static void insert(BST parent, int val) {
    if (val > parent.val) {
      if (parent.right != null)
        insert(parent.right, val);
      else
	parent.right = new BST(parent, val);
    } else {
      if (parent.left != null)
        insert(parent.left, val);
      else
	parent.left = new BST(parent, val);
    }
  }

  public static void remove(int val) {
    BST node = find(val);
    if (node != null) {
      remove(node);
    }
  }
  
  public static void remove(BST node) {
    if (node.left == null)
      if (node.right != null)
        swap_child(node.parent, node, node.right);
      else
        swap_child(node.parent, node, null);
    else if (node.right == null)
      swap_child(node.parent, node, node.left);
    else {
      // Both children present, copy over left child
      node.val = node.left.val;
      node.left = node.left.left;
      node.right = node.left.right;
      remove(node.left);
    }
  }

  private static void swap_child(BST parent, BST old_child, BST new_child) {
    if (new_child != null)
      new_child.parent = parent;
    if (parent == null)
      root = new_child;
    else if (parent.left == old_child)
      parent.left = new_child;
    else
      parent.right = new_child;
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

  public static void print() {
    if (root == null)
      System.out.println("empty");
    else
      print(root, 0);
  }

  public static void print(BST node, int depth) {
    if (node == null)
      return;

    System.out.println("depth: " + depth + ", " + node.toString());
    print(node.left, depth + 1);
    print(node.right, depth + 1);
  }

  public static void main(String[] args) {
    //System.out.println("........Empty........");
    insert(100);
    insert(110);
    insert(120);
    insert(115);
    print();
  }
}
