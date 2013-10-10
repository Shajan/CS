import java.io.*;

class Test {
    public static void main(String args[]) {
        String fileName = "data";
        Serialization s = new Serialization();
        Init(s);

/*
        System.out.println("-------Serialization:Original object");
        s.print();
        ReadWrite(s, fileName + ".ser");
*/

        External e = new External();
        Init(e);
/*
        System.out.println("-------Externalization:Original object");
        e.print();
        ReadWrite(e, fileName + ".ext");
*/
        int n = 1000000;

        Serialization sa[] = new Serialization[n];
        External ea[] = new External[n];

        for (int i=0; i<n; ++i) {
            sa[i] = s;
            ea[i] = e;
        }

        long serTime = PerfTest(sa, fileName + ".ser");
        long extTime = PerfTest(ea, fileName + ".ext");

        if (extTime > serTime) {
            System.out.println("Serialization is faster by (ns): " + (extTime - serTime));
            System.out.println("Serialization is faster by (%): " + 100*(extTime - serTime)/extTime);
        } else {
            System.out.println("Externalization is faster by (ns): " + (serTime - extTime));
            System.out.println("Serialization is faster by (%): " + 100*(serTime - extTime)/serTime);
        }
    }

    private static void Init(Record r) {
        r.type = "faces"; r.x = 500; r.y = 300; r.w = 68; r.h = 72;
    }

    private static long PerfTest(Record r[], String fileName) {
        long timetowrite = 0;
        long timetoread = 0;
        try
        {
            FileOutputStream fileOut = new FileOutputStream(fileName);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            timetowrite = System.nanoTime();
            for (int i=0; i<r.length; ++i)
                out.writeObject(r[i]);
            timetowrite = System.nanoTime() - timetowrite;
            out.close();
            fileOut.close();

            FileInputStream fileIn = new FileInputStream(fileName);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            timetoread = System.nanoTime();
            for (int i=0; i<r.length; ++i)
                r[i] = (Record) in.readObject(); 
            timetoread = System.nanoTime() - timetoread;
            in.close();
            fileIn.close();
            System.out.println("--------- " + fileName);
            System.out.println("Read time (ns)" + timetoread);
            System.out.println("Write time (ns)" + timetowrite);
        } catch(IOException i) {
            i.printStackTrace();
            return 0;
        } catch(ClassNotFoundException c) {
            System.out.println("class not found");
            c.printStackTrace();
            return 0;
        }

        return timetowrite + timetoread;
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
    public int x;
    public int y;
    public int w;
    public int h;
    public String type;

    public Record() {}
 
    public void print() {
        System.out.println("x: " + x);
        System.out.println("y: " + y);
        System.out.println("w: " + w);
        System.out.println("h: " + h);
        System.out.println("Type: " + type);
    }
}

class Serialization extends Record implements Serializable {
}

class External extends Record implements Externalizable {
    public External() { super(); }
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(x);
        out.writeInt(y);
        out.writeInt(w);
        out.writeInt(h);
        out.writeObject(type);
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        x    = in.readInt();
        y    = in.readInt();
        w    = in.readInt();
        h    = in.readInt();
        type = (String) in.readObject();
    }
}
