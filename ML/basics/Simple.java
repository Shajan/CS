/*
 * Demonstrates key building blocks of Machine Learning
 *
 * Problem: Approximate rules of a fuction by observing
 * it's input/output
 *
 * Convention/Concepts:
 *   x : Input(s)
 *   y : Known output - also called labels
 *   y-predicted : Predicted output
 *   w : Weights, when combined with input produces output
 *       Goal of training is to discover these weights
 *   e : Error - difference between prediction and actual output
 *   model : A set of weights
 *   train : look at known input/output pairs then figure out weights
 *
 * How: See different techniques below
 */
class Simple {
  public static void main(String args[]) {
  }
}

interface ITrain {
  int[] weights(float pair(x, y)
}

interface IPredict {
  int[] predict(float x[], float w[]);
}

/*
 * 1) LinearModel
 * --------------
 * Assume output is a linear combination of input with weight
 * Start with a random value of w, then refine w by reducing error in prediciton
 *   y-pred = x * w + w-0
 *   e = y-pred - y
 */
