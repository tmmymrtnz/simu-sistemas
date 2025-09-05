package simulation;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        Map map = new Map(0.09);
        map.printMap();
        Map map2 = new Map(0.03);
        map2.printMap();
        Map map3 = new Map(0.05);
        map3.printMap();
        Map map4 = new Map(0.07);
        map4.printMap();
    }
}