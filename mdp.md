# Markov Decision Process

The lunar lander problem can be framed as the following MDP:
- **State space**. 
    1. The `x` coordinate of the shuttle (float)
    2. The `y` coordinate of the shuttle (float)
    3. The `x` velocity of the shuttle (float)
    4. The `y` velocity of the shuttle (float)
    5. The orientation angle of the shuttle (float)
    6. The angular velocity of the shuttle (float)
    7. Is the left leg in contact with the ground? (bool)
    8. Is the right leg in contact with the ground? (bool)
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
