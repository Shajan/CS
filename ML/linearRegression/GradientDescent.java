import java.util.Arrays;

/*
 * Intution behind why this works: theta = theta - (alpha/m) * (x*theta' - y)' * x
 */
class GradientDescent {
  static boolean debugAllIterations = false;
  static boolean debugSampleIterations = false;

  public static void main(String args[]) {
    singleFeature();
    multipleLabelsSingleFeature();
  }

  /*
   * Finding a weight such that error is minimized, one feature
   */
  static void singleFeature() {
  /*
   * feature : input value
   * label : output value - function(feature), try different function
   *
   * theta : weight that needs to be tuned, so that 'feature * theta ==> output'
   * hypothesis : feature * theta
   * alpha : learning rate
   * error : hypothesis - label
   *
   * linear regression: do till error is small
   *   theta <= theta - alpha * error
   */

    double feature = 100.0;

    // Behavior with different values for alpha
    train(0.1, feature, feature*10, "Multiply by 10 : ");        // Fails to converge
    train(0.01, feature, feature*10, "Multiply by 10 : ");       // Fails to converge
    train(0.001, feature, feature*10, "Multiply by 10 : ");      // Fails to converge
    train(0.0001, feature, feature*10, "Multiply by 10 : ");     // Converges in 1 iteration
    train(0.00001, feature, feature*10, "Multiply by 10 : ");    // Converges in 65 iteration
    train(0.000001, feature, feature*10, "Multiply by 10 : ");   // Converges in 677 iteration

    // Different operations
    train(0.00001, feature, feature*100, "Multiply by 100 : ");  // 88 iterations
    train(0.00001, feature, feature/10, "Divide by 10 : ");      // 43 iterations
    train(0.00001, feature, feature/(-1), "Multiply by -1 : ");  // 51 iterations
    train(0.00001, feature, feature/(-2), "Divide by -2 : ");    // 48 iterations
  }

  /*
   * Find alpha, where f(x) = alpha*x
   *   This won't work with non linear relations
   */
  static double train(double alpha, double feature, double label, String note) {
    int iterations = 0;
    double theta = 1.0;
    double error, hypothesis;

    if (debugAllIterations)
      System.out.println(String.format("%s Alpha %g, feature %.5f, label %.5f", note, alpha, feature, label));

    do {
      hypothesis = feature * theta;
      error = hypothesis - label;
      if (error < 1 && error > -1) {
        System.out.println(String.format("%s theta %.5f, iterations %d", note, theta, iterations));
        if (debugAllIterations)
          System.out.println(String.format("\tAlpha %g, feature %.5f, label %.5f", alpha, feature, label));
        return theta;
      }
      theta -= alpha * error * feature;
      ++iterations;
      if (debugAllIterations || (debugSampleIterations && (iterations % 1000 == 0)))
        System.out.println(String.format("\ttheta %f, error %f", theta, error));
    } while (iterations <= 1000);

    System.out.println(String.format("%s Alpha %g, feature %.5f, label %.5f, iterations %d, theta %.5f [Failed to converge]",
      note, alpha, feature, label, iterations, theta));
    return theta;
  }

  static double[] multiply(double x, double input[]) {
    double result[] = new double[input.length];
    for (int i=0; i<input.length; ++i) {
      result[i] = input[i] * x;
    }
    return result;
  }

  static void multipleLabelsSingleFeature() {
    double feature[] = { 0.1, 0, 1.0, 10.0, 100, -0.5, -1.0, 100000 };
    double label[] = new double[feature.length];
    double alpha = 0.0000000001; 

    train(alpha, feature, multiply(1, feature), "Multiply by 1");
    train(alpha, feature, multiply(-1, feature), "Multiply by -1");
    train(alpha, feature, multiply(-100, feature), "Multiply by -100");
    train(alpha, feature, multiply(0.01, feature), "Multiply by 0.01");
    train(alpha, feature, multiply(-0.001, feature), "Multiply by -0.001");
    train(alpha, feature, multiply(1000, feature), "Multiply by 1000");
  }

  static double train(double alpha, double feature[], double label[], String note) {
    int iterations = 0;
    int m = label.length; // number of labels 'm'
    double theta = 1.0;
    double error[] = new double[m];

    if (debugAllIterations)
      System.out.println(String.format("%s Alpha %g, feature %s, label %s", note, alpha, toStr(feature), toStr(label)));

    do {
      // Compute error for the current values of theta
      double sumSqError = 0.0;
      double sumErrorTimesFeature = 0.0;
      for (int i=0; i<m; ++i) {
        double e = feature[i] * theta - label[i];
        error[i] = e;
        sumErrorTimesFeature += e*feature[i];
        sumSqError += Math.pow(e, 2);
      }
      double avgError = sumSqError/m;

      if (avgError < 0.01) {
        System.out.println(String.format("%s theta %f, iterations %d" note, theta, iterations));
        if (debugAllIterations)
          System.out.println(String.format("\tAlpha %g, feature %s, label %s", alpha, toStr(feature), toStr(label)));
        return theta;
      }

      // Update theta
      theta -= alpha*(sumErrorTimesFeature/m);
      if (Double.isNaN(theta))
        break;

      ++iterations;
      if (debugAllIterations || (debugSampleIterations && (iterations % 100 == 0)))
        System.out.println(String.format("\ttheta %.8f, error %s", theta, toStr(error)));
    } while (iterations <= 1000000);

    System.out.println(String.format("%s Alpha %g, feature %s, label %s, iterations %d, theta %.5f [Failed to converge]",
      note, alpha, toStr(feature), toStr(label), iterations, theta));
    return theta;
  }

  static String toStr(double d[]) {
    StringBuilder sb = new StringBuilder();
    sb.append("[");
    for (int i=0; i<d.length-1; ++i) {
      sb.append(String.format("%04.04f,", d[i]));
    }
    sb.append(String.format("%04.04f]", d[d.length-1]));
    return sb.toString();
  }
}
