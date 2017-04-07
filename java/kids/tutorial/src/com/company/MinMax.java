package com.company;

/**
 * Created by sdasan on 4/7/17.
 */
public class MinMax {
    public static void test() {
        /*
        System.out.println("max of 10, 15 is " + max(10, 15));
        System.out.println("max of 15, 10 is " + max(15, 10));
        System.out.println("max of 10, 10 is " + max(10, 10));

        System.out.println("min of 10, 15 is " + min(10, 15));
        System.out.println("min of 15, 10 is " + min(15, 10));
        System.out.println("min of 10, 10 is " + min(10, 10));

        System.out.println("max of 10, 15, 20 is " + max(10, 15, 20));
        System.out.println("max of 10, 20, 15 is " + max(10, 20, 15));
        System.out.println("max of 20, 15, 10 is " + max(20, 15, 10));
        System.out.println("max of 10, 10, 20 is " + max(10, 10, 20));
        System.out.println("max of 10, 20, 20 is " + max(10, 20, 20));
        System.out.println("max of 8, 2, 10 is " + max(8, 2, 10));
        System.out.println("max of 100, 1, 5 is " + max(100, 1, 5));

        System.out.println("min of 10, 15, 20 is " + min(10, 15, 20));
        System.out.println("min of 15, 10, 20 is " + min(15, 10, 20));
        System.out.println("min of 20, 10, 15 is " + min(20, 10, 15));
        System.out.println("min of 20, 15, 10 is " + min(20, 15, 10));
        System.out.println("min of 10, 5, 10 is " + min(10, 5, 10));
        System.out.println("min of 10, 10, 20 is " + min(10, 10, 20));
        System.out.println("min of 10, 10, 10 is " + min(10, 10, 10));
        */

        System.out.println(max(1,2,3));
        System.out.println(max(1,3,2));
        System.out.println(max(2,3,1));
        System.out.println(max(2,1,3));
        System.out.println(max(3,1,2));
        System.out.println(max(3,2,1));
    }

    static int max(int a, int b) {
        if (a > b)
            return a;
        else
            return b;
    }

    static int min(int a, int b) {
        if (a < b)
            return a;
        else
            return b;
    }

    static int max(int a, int b, int c) {
        if (a > c) {
            return a;
        }
        if (c > b)
            return c;
        else
            return b;
    }


    static int min(int a, int b, int c) {
        if (a < b)
            return a;
        if (c < b)
            return c;
        else
            return b;
    }

}

