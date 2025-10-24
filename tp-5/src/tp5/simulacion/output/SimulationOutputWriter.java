package tp5.simulacion.output;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.SimulationConfig;
import tp5.simulacion.model.PedestrianDynamicsModel;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Handles writing the simulation state and contact events to plain text files.
 *
 * File formats (space separated):
 *   states.txt  : step time agentId x y vx vy ax ay radius
 *   contacts.txt: ordinal time agentId
 *
 * The header includes model and configuration metadata for traceability.
 */
public class SimulationOutputWriter implements AutoCloseable {
    private final BufferedWriter stateWriter;
    private final BufferedWriter contactWriter;
    private final List<String> runMetadata;

    public SimulationOutputWriter(
            Path outputDir,
            SimulationConfig config,
            PedestrianDynamicsModel model,
            List<String> runMetadata
    ) throws IOException {
        Files.createDirectories(outputDir);

        Path statesPath = outputDir.resolve("states.txt");
        Path contactsPath = outputDir.resolve("contacts.txt");

        this.stateWriter = Files.newBufferedWriter(statesPath);
        this.contactWriter = Files.newBufferedWriter(contactsPath);
        this.runMetadata = runMetadata;

        writeHeaders(config, model);
    }

    private void writeHeaders(SimulationConfig config, PedestrianDynamicsModel model) throws IOException {
        List<String> modelMeta = model.describeParameters();

        stateWriter.write("# Simulation state log");
        stateWriter.newLine();
        stateWriter.write("# domainSize=" + config.domainSize());
        stateWriter.write(" periodic=" + config.periodic());
        stateWriter.write(" interactionRadius=" + config.interactionRadius());
        stateWriter.write(" dt=" + config.timeStep());
        stateWriter.write(" outputInterval=" + config.outputInterval());
        stateWriter.write(" totalSteps=" + config.totalSteps());
        if (!modelMeta.isEmpty()) {
            stateWriter.write(" modelParams=" + String.join(",", modelMeta));
        }
        if (runMetadata != null && !runMetadata.isEmpty()) {
            stateWriter.write(" runMeta=" + String.join(",", runMetadata));
        }
        stateWriter.newLine();
        stateWriter.write("step time agentId x y vx vy ax ay radius");
        stateWriter.newLine();

        contactWriter.write("# Unique contacts between mobile agents and the central obstacle");
        contactWriter.newLine();
        contactWriter.write("ordinal time agentId");
        contactWriter.newLine();
    }

    public void writeState(long step, double time, List<Agent> agents, List<double[]> accelerations) throws IOException {
        for (int i = 0; i < agents.size(); i++) {
            Agent agent = agents.get(i);
            double[] acc = accelerations.get(i);
            stateWriter.write(step + " " + time + " " + agent.getId() + " "
                    + agent.getPosition().x() + " " + agent.getPosition().y() + " "
                    + agent.getVelocity().x() + " " + agent.getVelocity().y() + " "
                    + acc[0] + " " + acc[1] + " "
                    + agent.getRadius());
            stateWriter.newLine();
        }
    }

    public void writeContact(long ordinal, double time, int agentId) throws IOException {
        contactWriter.write(ordinal + " " + time + " " + agentId);
        contactWriter.newLine();
    }

    @Override
    public void close() throws IOException {
        stateWriter.flush();
        stateWriter.close();
        contactWriter.flush();
        contactWriter.close();
    }
}
