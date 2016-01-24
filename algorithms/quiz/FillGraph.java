/**
  X
  X o o X
  X X X X o X
  X X X X X X
  4 2 2 3 1 2= 3
  
  X       X
  X   X   X 
  X X X X X
  
  X   *   X
  X   X   X 
  X X X X X
  
  Waterlevel = MyNum - Min(Max(left array), Max(Right array)) > 0) ?
 */


class Solution {
  public static void main(String[] args) {
    //int[] graph = {4, 1, 2, 4, 1, 2};
    int[] graph = {5, 1, 2, 1, 5};
    int[] rightMax = new int[graph.length];
    int volume = 0;
    
    int maxSoFar = -1;
    for (int i=graph.length - 1; i>=0; --i) {
      if (graph[i] > maxSoFar) 
        maxSoFar = graph[i];
      rightMax[i] = maxSoFar;
    }
    
    int leftMax = -1;
    for (int i=0; i<graph.length; ++i) {
      int max = Math.min(leftMax, rightMax[i]);
      if (max > 0 && graph[i] < max) {
        volume += max - graph[i]; 
      }
      if (graph[i] > leftMax) 
        leftMax = graph[i];
    }
    System.out.println(volume);
  }
    /*
    for (int i=0; i<graph.length; ++i) {
      int level = Math.min(max(graph, 0, i-1), max(graph, i+1, end));
      if (level > 0 && (level > graph[i])) {
          volume += (level - graph[i]);
          System.out.println(String.format("Adding %d",graph[i] - level));
      }
    }
  static int max(int[] a, int start, int end) {
   System.out.println(String.format("[%d %d]", start, end));
    if (start < 0 || end >= a.length || start > end)
      return -1;
   int m = -1; // current max
   for (int i=start; i<=end; ++i) {
     if (a[i] > m)
       m = a[i];
   }
   System.out.println(String.format("%d, [%d %d]", m, start, end));
   return m;
  }
    */
}