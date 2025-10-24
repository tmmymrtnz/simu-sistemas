package tp5.simulacion.engine;

import tp5.simulacion.core.Agent;
import tp5.simulacion.core.Vector2;

import java.util.HashMap;
import java.util.Map;

/**
 * Keeps track of which agents already triggered a contact event and resets the state when they wrap.
 */
public class ContactTracker {
    private final Map<Integer, Boolean> agentHasContacted = new HashMap<>();

    public boolean shouldRegisterContact(Agent agent, Vector2 centralPosition, double centralRadius) {
        double thr = agent.getRadius() + centralRadius;
        double dx = agent.getPosition().x() - centralPosition.x();
        double dy = agent.getPosition().y() - centralPosition.y();
        double distSq = dx * dx + dy * dy;

        boolean alreadyCounted = agentHasContacted.getOrDefault(agent.getId(), false);
        boolean touches = distSq <= thr * thr;
        if (touches && !alreadyCounted) {
            agentHasContacted.put(agent.getId(), true);
            return true;
        }
        return false;
    }

    public void resetIfWrapped(Agent agent, boolean wrapped) {
        if (wrapped) {
            agentHasContacted.put(agent.getId(), false);
        }
    }
}
