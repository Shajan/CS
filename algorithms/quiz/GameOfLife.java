import java.io.Console;
import java.util.*;

class GameOfLife {
  private int rows, cols;
  private byte[][] state;
  private int iteration;

  private GameOfLife(int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    this.state = new byte[2][];
    this.state[0] = new byte[(int)Math.ceil((double)(rows*cols)/8)];
    this.state[1] = new byte[(int)Math.ceil((double)(rows*cols)/8)];
  }

  private byte[] live() { return state[iteration%2]; }
  private byte[] next() { return state[(iteration + 1)%2]; }

  private int offset(int row, int col) { return rows*row + col; }
  private int index(int offset) { return offset/8; }
  private byte mask(int offset) { return (byte)((byte)1 << (offset%8)); }

  private boolean isLive(int row, int col) {
    byte[] s = live();
    int offs = offset(row, col);
    return ((s[index(offs)] & mask(offs)) != 0);
  }

  private void setLive(int row, int col) {
    byte[] s = next();
    int offs = offset(row, col);
    int idx = index(offs);
    s[idx] |= mask(offs);
  }

  private void setDead(int row, int col) {
    byte[] s = next();
    int offs = offset(row, col);
    int idx = index(offs);
    s[idx] &= ~mask(offs);
  }

  private boolean isValid(int row, int col) {
    return (row >= 0 && row < rows && col >= 0 && col < cols);
  }

  private int neighbours(int row, int col) {
    int count = 0;
    for (int i=row-1; i<=row+1; ++i) {
      for (int j=col-1; j<=col+1; ++j) {
        if (i == row && j == col)
          continue;
        if (isValid(i, j) && isLive(i, j))
          ++count;
      }
    }
    return count;
  }

  private void advance() {
    for (int i=0; i<rows; ++i) {
      for (int j=0; j<cols; ++j) {
        int count = neighbours(i, j);
        if (isLive(i, j)) {
          if (count < 2 || count > 3)
            setDead(i, j);
          else
            setLive(i, j);
        } else {
          if (count == 3)
            setLive(i, j);
          else
            setDead(i, j);
        }
      }
    }
    ++iteration;
  }

  private void initCursor() {
    System.out.print(String.format("%c[%d;%df", 0x1B, 0, 0));
  }

  private void print() {
    initCursor();
    System.out.println(iteration);
    for (int i=0; i<rows; ++i) {
      System.out.print(String.format("%2d:[", i));
      for (int j=0; j<cols; ++j) {
        System.out.print(isLive(i,j) ? "O" : ".");
      }
      System.out.println("]");
    }
  }

  private void trace() {
    print();
    try {
      Thread.sleep(100);
    } catch (InterruptedException e) {
    }
    //console.readLine();
  }

  private void init() {
    Random rand = new Random();
    for (int i=0; i<rows; ++i)
      for (int j=0; j<cols; ++j)
        if (rand.nextInt(4) == 0)
          setLive(i, j);
    iteration = 1;
  }

  public static void main(String[] args) {
    Console console = System.console();
    GameOfLife gol = new GameOfLife(35, 120);
    gol.init();
    gol.trace();
    for (int i=0; i<100; ++i) {
      gol.advance();
      gol.trace();
    }
  }
}
