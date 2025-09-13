package simulation;

/** Constantes globales de la simulación. */
public final class Constants {
    private Constants() {}

    /** Tolerancia temporal/espacial y micro-separación (paso mínimo de eventos). */
    public static final double TIME_EPS = 1e-7;

    /** Tolerancia relativa para considerar P_left ~ P_right (estado estacionario). */
    public static final double PRESSURE_TOL = 0.1; // 5%

    /** Cantidad de muestras consecutivas cumpliendo la tolerancia. */
    public static final int STEADY_SAMPLES = 100;

    /** Umbral mínimo de presión media para evitar estacionario "falso" con ceros. */
    public static final double MIN_MEAN_PRESSURE = 1e-8; // ajustable

    /** Δt fijo del muestreo de presión (seg). */
    public static final double PRESSURE_SAMPLE_DT = 5.0; 
}
