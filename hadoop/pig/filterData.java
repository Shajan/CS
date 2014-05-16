import java.io.IOException;
import org.apache.pig.PigServer;

public class filterData { 
  public static void main(String[] args) {
    try {
      PigServer pigServer = new PigServer("local");
      // Change for mapreduce 
      // = new PigServer("mapreduce");
      firstCol(pigServer, "data.csv");
    } catch(Exception e) {
    }
 }
 public static void firstCol(PigServer pigServer, String inputFile) throws IOException {
    pigServer.registerQuery("A = load '" + inputFile + "' using PigStorage(',');");
    pigServer.registerQuery("B = foreach A generate $0 as col1;");
    pigServer.store("B", "col1.out");
 }
}
