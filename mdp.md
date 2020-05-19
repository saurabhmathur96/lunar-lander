# Markov Decision Process

The lunar lander problem can be framed as the following MDP:
- **State space**. The `(x, y)` coordinates of the shuttle.
- **Action space**. 4 discrete actions are available:
    1. Do Nothing
    2. Fire left orientation engine
    3. Fire main engine
    4. Fire right orientation engine
- **Transitions**
   1. Do Nothing: Downward movement(gravity)
   2. Fire left/right orientation engine: Diagonal movement
   3. Fire main engine: Upward movement
- **Reward**.
    1. Crash: -100
    2. Fire main engine: -0.3/frame
    3. Land in landing zone: 200
    4. Leg-Ground contact: 10
