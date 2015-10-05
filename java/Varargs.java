/*
 * javac Varargs.java
 * java Varargs
 */
class Varargs {
    public static void main(String args[]) {
        var_arg(1, "abc", "def");
        var_arg(1, new Object[]{"abc", "def"});
    }
 
    public static void var_arg(int i, Object... args) {
        System.out.println(i);
        for (Object arg: args) {
            System.out.println(arg); 
        }
    }
}

