/*
 * Intution behind why this works: theta = theta - (alpha/m) * (x*theta' - y)' * x
 */
class GradientDescent {
  static boolean debugAllIterations = false;
  static boolean debugSampleIterations = false;

  public static void main(String args[]) {
    singleFeature();
  }

  /*
   * Finding a weight such that error is minimized, one feature
   */
  public static void singleFeature() {
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

  public static double train(double alpha, double feature, double label, String note) {
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
}
