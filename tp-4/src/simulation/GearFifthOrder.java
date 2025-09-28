package simulation;

public class GearFifthOrder implements  NumericMethod{

    private double[] y = new double[6]; //derivadas

    private final AccelerationFunction accelerationFunction;

    private static final double[] alphas = {
            3.0/16.0,
            251.0/360.0,
            1.0,
            11.0/18.0,
            1.0/6.0,
            1.0/60.0
    };

    public GearFifthOrder(AccelerationFunction accelerationFunction, double x0, double v0, double t0) {
        this.accelerationFunction = accelerationFunction;

        // Inicialización de derivadas
        y[0] = x0;
        y[1] = v0;
        y[2] = accelerationFunction.computeAcceleration(x0, v0, t0);
        y[3] = 0.0;
        y[4] = 0.0;
        y[5] = 0.0;
    }


    @Override
    public State solve(Double x, Double v, Double t, Double dt) {
        // --- 1. Predictor ---
        double[] yp = new double[6];
        for (int k = 0; k < 6; k++) {
            double sum = 0.0;
            for (int j = 0; j < 6 - k; j++) {
                sum += Math.pow(dt, j) / factorial(j) * y[k + j];
            }
            yp[k] = sum;
        }

        // --- 2. Evaluar aceleración real en t+dt ---
        double aEval = accelerationFunction.computeAcceleration(yp[0], yp[1], t + dt);

        // --- 3. Diferencia en aceleración ---
        double Delta = aEval - yp[2];

        // --- 4. Corrección ---
        double[] yc = new double[6];
        for (int k = 0; k < 6; k++) {
            double factor;
            if (k >= 2) {
                factor = Math.pow(dt, k - 2) / factorial(k - 2);
            } else if (k == 0) {
                factor = (dt * dt) / 2.0;
            } else { // k == 1
                factor = dt;
            }
            yc[k] = yp[k] + alphas[k] * Delta * factor;
        }

        // --- 5. Actualizar derivadas para el próximo paso ---
        y = yc;

        // Devolver el nuevo estado
        return new State(y[0], y[1]);
    }

    // factorial pequeño
    private static int factorial(int n) {
        if (n < 2) return 1;
        int f = 1;
        for (int i = 2; i <= n; i++) f *= i;
        return f;
    }
}
