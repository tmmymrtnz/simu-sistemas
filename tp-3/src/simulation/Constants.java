package simulation;

/** Constantes globales de la simulación. */
public final class Constants {
    private Constants() {}

    /** Tolerancia temporal/espacial y micro-separación (paso mínimo de eventos). */
    public static final double TIME_EPS = 1e-7;

    /** Tolerancia relativa para considerar P1 ~ P2 (estado estacionario). */
    public static final double PRESSURE_TOL = 0.02; // 2%

    /** Cantidad de muestras consecutivas cumpliendo la tolerancia. */
    public static final int STEADY_SAMPLES = 10;
}
