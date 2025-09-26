package simulation;

public interface NumericMethod {

    State solve(Double x0, Double v0, Double t, Double dt);
}
