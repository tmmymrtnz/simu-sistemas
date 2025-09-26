package simulation;

public class State {
    private final double x;
    private final double v;

    public State(double x, double v) {
        this.x = x;
        this.v = v;
    }

    public double getX() { return x; }
    public double getV() { return v; }
}