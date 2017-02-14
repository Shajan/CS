import java.util.Arrays;

/*
 * Intution behind why this works: theta = theta - (alpha/m) * (x*theta' - y)' * x
 */
class GradientDescent {
  static boolean debugAllIterations = false;
  static boolean debugSampleIterations = true;

  public static void main(String args[]) {
    //singleFeature();
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

    // Multiply by a constant
    train(0.1, feature, feature*10, "Multiply by 10");
    train(0.01, feature, feature*10, "Multiply by 10");
    train(0.001, feature, feature*10, "Multiply by 10");

    // Divide by a constant
    train(0.1, feature, feature/10, "Divide by 10");
    train(0.01, feature, feature/10, "Divide by 10");
    train(0.001, feature, feature/10, "Divide by 10");
  }

  static double train(double alpha, double feature, double label, String note) {
    int iterations = 0;
    double theta = 1.0;
    double error, hypothesis;
    System.out.println(String.format("%s Alpha %.5f, feature %.5f, label %.5f", note, alpha, feature, label));
    do {
      hypothesis = feature * theta;
      error = hypothesis - label;
      if (error < 1 && error > -1) {
        System.out.println(String.format("%s Alpha %.5f, feature %.5f, label %.5f, iterations %d, theta %.5f",
            note, alpha, feature, label, iterations, theta));
        return theta;
      }
      theta = theta - alpha * error;
      ++iterations;
      if (debugAllIterations || (debugSampleIterations && (iterations % 100 == 0)))
        System.out.println(String.format("\ttheta %f, error %f", theta, error));
    } while (iterations <= 1000);

    System.out.println(String.format("%s Alpha %.5f, feature %.5f, label %.5f, iterations %d, theta %.5f [Failed to converge]",
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
    double feature[] = { 0.1, 0, 1.0, 10.0, 100, -0.5, -1.0, 10000 };
    double label[] = new double[feature.length];

    train(0.0001, feature, multiply(1, feature), "Multiply by 1");
/*
    train(0.001, feature, multiply(-1, feature), "Multiply by -1");
    train(0.001, feature, multiply(-100, feature), "Multiply by -100");
    train(0.001, feature, multiply(0.01, feature), "Multiply by 0.01");
    train(0.001, feature, multiply(-0.001, feature), "Multiply by -0.001");
    train(0.001, feature, multiply(1000, feature), "Multiply by 1000");
*/
  }

  static double[] train(double alpha, double feature[], double label[], String note) {
    int iterations = 0;
    int m = label.length; // number of labels 'm'
    double theta[] = new double[m];
    double error[] = new double[m];

    System.out.println(
      String.format("%s Alpha %.5f, feature %s, label %s", note, alpha, toStr(feature), toStr(label)));

    do {
      // Compute error for the current values of theta
      double sumSqError = 0.0;
      for (int i=0; i<m; ++i) {
        double e = feature[i] * theta[i] - label[i];
        error[i] = e; 
        sumSqError += Math.pow(e, 2); 
      }
      double avgError = sumSqError/m; 

      if (avgError < 0.01) {
        System.out.println(String.format("%s Alpha %.5f, feature %s, label %s, iterations %d, theta %s",
            note, alpha, toStr(feature), toStr(label), iterations, toStr(theta)));
        return theta;
      }

      // Update theta
      for (int i=0; i<m; ++i)
        theta[i] = theta[i] - alpha * error[i];

      ++iterations;
      if (debugAllIterations || (debugSampleIterations && (iterations % 100 == 0)))
        System.out.println(String.format("\ttheta %s, error %s", toStr(theta), toStr(error)));
    } while (iterations <= 1000000);

    System.out.println(String.format("%s Alpha %.5f, feature %s, label %s, iterations %d, theta %s [Failed to converge]",
      note, alpha, toStr(feature), toStr(label), iterations, toStr(theta)));
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
