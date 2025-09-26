package simulation;

public class GearFifthOrder implements  NumericMethod{


    @Override
    public State solve(Double x0, Double v0, Double t, Double dt) {
        return new State(x0,v0);
    }
}
