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

    public GearFifthOrder(AccelerationFunction accelerationFunction, double x0, double v0, double t0, double k, double m, double gamma) {
        this.accelerationFunction = accelerationFunction;
        // Inicialización de derivadas
        y[0] = x0;
        y[1] = v0;
        y[2] = accelerationFunction.computeAcceleration(x0, v0, t0);
        y[3] = (-k)/m * v0 - (gamma/m) * y[2]; // derivada de la aceleración (jerk)
        y[4] = (-k)/m * y[2] - (gamma/m) * y[3]; // segunda derivada de la aceleración
        y[5] = (-k)/m * y[3] - (gamma/m) * y[4]; // tercera derivada de la aceleración
    }




    @Override
    public State solve(Double x, Double v, Double t, Double dt) {
        // --- 1. Predictor ---

        double[] yp = new double[6];

        yp[0] = y[0] + y[1] * dt + y[2] * dt * dt / 2.0 + y[3] * dt * dt * dt / 6.0 + y[4] * dt * dt * dt * dt / 24.0 + y[5] * dt * dt * dt * dt * dt / 120.0;
        yp[1] = y[1] + y[2] * dt + y[3] * dt * dt / 2.0 + y[4] * dt * dt * dt / 6.0 + y[5] * dt * dt * dt * dt / 24.0;
        yp[2] = y[2] + y[3] * dt + y[4] * dt * dt / 2.0 + y[5] * dt * dt * dt / 6.0;
        yp[3] = y[3] + y[4] * dt + y[5] * dt * dt / 2.0;
        yp[4] = y[4] + y[5] * dt;
        yp[5] = y[5];


        // --- 2. Evaluar aceleración real en t+dt ---
        double aEval = accelerationFunction.computeAcceleration(yp[0], yp[1], t + dt);

        // --- 3. Diferencia en aceleración ---
        double deltaA = aEval - yp[2];
        double deltaR2 = deltaA * dt * dt / 2.0;

        // --- 4. Corrección ---
        double[] yc = new double[6];
        
        // Corrección usando deltaA directamente con factores correctos de Gear-5
        yc[0] = yp[0] + alphas[0] * deltaR2;                              // posición
        yc[1] = yp[1] + alphas[1] * deltaR2 / dt;                         // velocidad
        yc[2] = yp[2] + alphas[2] * deltaR2 * 2.0 / (dt * dt);          // aceleración
        yc[3] = yp[3] + alphas[3] * deltaR2 * 6.0 / (dt * dt * dt);      // derivada de la aceleración (jerk)
        yc[4] = yp[4] + alphas[4] * deltaR2 * 24.0 / (dt * dt * dt * dt); // segunda derivada de la aceleración
        yc[5] = yp[5] + alphas[5] * deltaR2 * 120.0 / (dt * dt * dt * dt * dt); // tercera derivada de la aceleración

        // --- 5. Actualizar derivadas para el próximo paso ---
        System.arraycopy(yc, 0, y, 0, 6);

        // Devolver el nuevo estado
        return new State(yc[0], yc[1]);
    }

    // factorial pequeño
    private static int factorial(int n) {
        if (n < 2) return 1;
        int f = 1;
        for (int i = 2; i <= n; i++) f *= i;
        return f;
    }
}
