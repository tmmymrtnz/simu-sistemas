package simulation;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        // Construimos tu mapa (ejemplo con L=0.05) y paredes
        Map map = new Map(0.05);

        // Crear N partículas con direcciones aleatorias (en el “lado izquierdo” si querés: ajustá x)
        int N = 12;
        map.createParticles(N);

        // Correr simulación event-driven y loggear a events.log
        Path logPath = Path.of("events.log");

        // Ojo: la clase Map tuya guarda walls y particles:
        List<Agent> agents = map.particles;                 // ya pobladas
        List<Wall>  walls  = map.walls;                     // ya definidas

        TargetEventSim sim = new TargetEventSim(agents, walls);
        sim.run(/*tMax=*/10, logPath,false);

        System.out.println("Simulación finalizada. Log en: " + logPath.toAbsolutePath());
    }
}
