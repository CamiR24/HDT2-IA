import numpy as np
import matplotlib.pyplot as plt

'''1'''
#Definir tamaño grid
grid_size = 4
gamma = 0.9
epsilon = 1e-4

V = np.zeros((grid_size, grid_size)) #Inicializa V0

#Si hay estados terminales 
terminal_states = {
    (0, 3): 1,
    (1, 3): -1
}

for (i, j), reward in terminal_states.items():
    V[i, j] = reward

'''2'''
#Definir acciones
actions = {
    "U": (-1, 0), #up
    "D": (1, 0), #down
    "L": (0, -1), #left
    "R": (0, 1) #right
}

#Función para moverse
def move(state, action):
    i, j = state
    di, dj = actions[action]
    ni, nj = i + di, j + dj
    
    if 0 <= ni < 4 and 0 <= nj < 4:
        return (ni, nj)
    else:
        return state  # choca y se queda
    
#Acciones perpendiculares
def get_perpendicular(action):
    if action in ["U", "D"]:
        return ["L", "R"]
    else:
        return ["U", "D"]
    
##VALUE ITERATION
living_reward = -0.04
gamma = 0.9
epsilon = 1e-4

while True:
    delta = 0
    V_new = V.copy()
    
    for i in range(4):
        for j in range(4):
            
            state = (i, j)
            
            # Saltar estados terminales
            if state in terminal_states:
                continue
            
            best_value = -np.inf
            
            for action in actions:
                
                total = 0
                
                # Acción principal (0.8)
                next_state = move(state, action)
                total += 0.8 * (living_reward + gamma * V[next_state])
                
                # Perpendiculares (0.1 cada una)
                for perp in get_perpendicular(action):
                    next_state = move(state, perp)
                    total += 0.1 * (living_reward + gamma * V[next_state])
                
                best_value = max(best_value, total)
            
            V_new[state] = best_value
            delta = max(delta, abs(V_new[state] - V[state]))
    
    V = V_new
    
    if delta < epsilon:
        break

'''3'''
policy = np.empty((4,4), dtype=object) #matriz política

for i in range(4):
    for j in range(4):
        
        state = (i, j)
        
        if state in terminal_states:
            policy[i, j] = "T"
            continue
        
        best_action = None
        best_value = -np.inf
        
        for action in actions:
            
            total = 0
            
            # Acción principal
            next_state = move(state, action)
            total += 0.8 * (living_reward + gamma * V[next_state])
            
            # Perpendiculares
            for perp in get_perpendicular(action):
                next_state = move(state, perp)
                total += 0.1 * (living_reward + gamma * V[next_state])
            
            if total > best_value:
                best_value = total
                best_action = action
        
        policy[i, j] = best_action

'''4'''
#Convertir acciones a flechas
arrow_map = {
    "U": "↑",
    "D": "↓",
    "L": "←",
    "R": "→",
    "T": "T"
}

print("Política Óptima:\n")

for i in range(4):
    row = ""
    for j in range(4):
        row += arrow_map[policy[i, j]] + "  "
    print(row)

plt.figure()
plt.imshow(V, cmap="viridis")
plt.colorbar()
plt.title("Mapa de Calor de V*(s)")
plt.show()