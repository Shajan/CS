package com.company;

/**
 * Created by sdasan on 4/7/17.
 */
public class Loops {
    public static void test() {
        whileLoop();
    }

    private static void whileLoop() {
        //printAllNumbers(5);
        //printBetween(1,10);
        printAllEven(20);
     }

    private static void printAllNumbers(int a) {
        int i = 0;

        while (i < a) {
            System.out.println(i);
            i = i + 1;
        }
    }

    private static void printBetween(int start, int end) {
        int i=start + 1;

        while (i<end) {
            System.out.println(i);
            i = i + 1;
        }
    }

    private static void printAllEven(int a) {
        int i = 0;

        while (i < a) {
            System.out.println(i);
            i = i + 2;
        }
    }
}
