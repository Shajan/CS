import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class Stream {
  public static void main(String args[]) {
    basic();
    advanced();
  }

  private static void basic() {

    // 0 to 4
    IntStream.range(0, 5)
      .forEach(System.out::println); 

    // 0, 2, 4
    IntStream.iterate(0, i -> i + 2)
      .limit(3) 
      .forEach(System.out::println);

    // 5 random numbers
    IntStream is = IntStream
      .generate(() -> { return (int)(Math.random() * 10000); });

    is.limit(5)
      .forEach(System.out::println);

    // to List
    List<Integer> l = IntStream.range(0, 5)
      .boxed()
      .collect(Collectors.toList());
    System.out.println(l);

    // to Array
    int[] a = IntStream.of(1, 2, 3, 4, 5)
      .toArray();
    System.out.println(a);

    // Filter
    List<Integer> numbers = List.of(1, 2, 3, 4, 5, 6);
    List<Integer> even = numbers
      .stream()
      .filter(x -> x % 2 == 0)
      .collect(Collectors.toList());
    System.out.println(even);  // [2, 4, 6]
  }

  private  static void advanced() {
    List<String> vowels = List.of("a", "e", "i", "o", "u");

    // sequential stream - nothing to combine
    StringBuilder sb1 = vowels
      .stream()
      .collect(
        StringBuilder::new,
        (x, y) -> x.append(y),                // StringBuilder.append
        (a, b) -> a.append(",").append(b));   // This won't get called!
    System.out.println(sb1.toString());

    // parallel stream - combiner is combining partial results
    StringBuilder sb2 = vowels
      .parallelStream()
      .collect(
        StringBuilder::new,
        (x, y) -> x.append(y),
        (a, b) -> a.append(",").append(b));
    System.out.println(sb2.toString());

    // Shorter version
    StringBuilder sb3 = vowels
      .parallelStream()
      .collect(StringBuilder::new, StringBuilder::append, StringBuilder::append);
    System.out.println(sb3.toString());
  }
}
