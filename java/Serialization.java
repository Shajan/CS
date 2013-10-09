import java.io.*;

class Test {
    public static void main(String args[]) {
        String fileName = "data";
        Serialization s = new Serialization();
            s.name = "Shajan Dasan";
            s.address = "Sammamish, WA";
            s.SSN = 123456789;
            s.number = 747;
        System.out.println("-------Serialization:Original object");
        s.print();
        ReadWrite(s, fileName + ".ser");

        External e = new External();
            e.name = s.name;
            e.address = s.address;
            e.SSN = s.SSN;
            e.number = s.number;
        System.out.println("-------Externalization:Original object");
        e.print();
        ReadWrite(e, fileName + ".ext");
    }

    private static void ReadWrite(Record r, String fileName) {
        try
        {
            FileOutputStream fileOut = new FileOutputStream(fileName);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(r);
            out.close();
            fileOut.close();

            FileInputStream fileIn = new FileInputStream(fileName);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            r = (Record) in.readObject(); 
            in.close();
            fileIn.close();

            System.out.println("\tFinal object");
            r.print();
        } catch(IOException i) {
            i.printStackTrace();
            return;
        } catch(ClassNotFoundException c) {
            System.out.println("class not found");
            c.printStackTrace();
            return;
        }
    }
}

class Record implements Serializable {
    public String name;
    public String address;
    public transient int SSN;
    public int number;

    public Record() {}
 
    public void print() {
        System.out.println("Name: " + name);
        System.out.println("Address: " + address);
        System.out.println("SSN: " + SSN);
        System.out.println("Number: " + number);
    }
}

class Serialization extends Record implements Serializable {
}

class External extends Record implements Externalizable {
    public External() { super(); }
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeObject(name);
        out.writeObject(address);
        out.writeInt(number);
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        name    = (String) in.readObject();
        address = (String) in.readObject();
        number  = in.readInt();
    }
}
