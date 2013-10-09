import java.io.*;

public class Serialization implements Serializable {
    private String name;
    private String address;
    private transient int SSN;
    private int number;
 
    private void print() {
        System.out.println("Name: " + name);
        System.out.println("Address: " + address);
        System.out.println("SSN: " + SSN);
        System.out.println("Number: " + number);
    }

    public static void main(String args[]) {
        String fileName = "/tmp/del/data/Serialization.ser";
        Serialization t = new Serialization();
        t.name = "Reyan Ali";
        t.address = "Phokka Kuan, Ambehta Peer";
        t.SSN = 11122333;
        t.number = 101;
        System.out.println("Original object");
        t.print();

        try
        {
            FileOutputStream fileOut = new FileOutputStream(fileName);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(t);

            FileInputStream fileIn = new FileInputStream(fileName);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            t = (Serialization) in.readObject(); 
            System.out.println("Final object");
            t.print();
        } catch(IOException i) {
            i.printStackTrace();
            return;
        } catch(ClassNotFoundException c) {
            System.out.println("Employee class not found");
            c.printStackTrace();
            return;
      }
    }
}

