package simulation;

/**
 * Holds kinetic and potential energy values for diagnostics.
 */
public class EnergySnapshot {
    private final double kinetic;
    private final double potential;

    public EnergySnapshot(double kinetic, double potential) {
        this.kinetic = kinetic;
        this.potential = potential;
    }

    public double getKinetic() {
        return kinetic;
    }

    public double getPotential() {
        return potential;
    }

    public double getTotal() {
        return kinetic + potential;
    }
}
