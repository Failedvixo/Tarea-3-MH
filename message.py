import numpy as np

#HIPERPARÁMETROS
# 1) Cant. partículas
# 2) Cant. Itt
# 3) Coef. Incercia (peso)
# 4) Coef. Congnitivo social
# 5)Velocidad
# Dominio (Pero viene dado)

class PSO:
    def __init__(self, num_particles, num_iterations, obj_func, dim=10, bounds=(0,1), inertia_weight = 0.4):
        self.num_particles = num_particles #1
        self.num_iterations = num_iterations #2
        self.obj_func = obj_func
        self.dim = dim 
        self.bounds = bounds
        self.inertia_weight = inertia_weight #3
        self.cognitive_coef = 1.5 #4.1
        self.social_coef = 1.5 #4.2
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dim)) #5
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(num_particles, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def train(self):
        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                score = self.obj_func(self.positions[i])
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
            
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                cognitive_velocity = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_coef * r2 * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                
                # Restricción de dominio
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
        
        return self.global_best_position, self.global_best_score


#Funciones objetivo

def f1(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def f2(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def f(x):
    d = len(x)
    return np.sum(-np.sin(x) * (np.sin((np.arange(1, d + 1) * x**2) / np.pi))**20)

f1_bounds = [-5.12, 5.12]
f2_bounds = [-30, 30]
f3_bounds = [0,np.pi]

weights = [0.7]
particles = [100]
results = []
pso = PSO(num_particles=50, num_iterations=5000, obj_func=f, bounds=f3_bounds, inertia_weight=0.7)
best_position, best_score = pso.train()

print(f"Mejor posición: {best_position}")
print(f"Mejor valor encontrado: {best_score}")    

"""
//grid_search por peso
for weight in weights:
    scores = []
    for _ in range(10):  # Perform 10 iterations for each weight
        pso = PSO(num_particles=50, num_iterations=10000, obj_func=f1, bounds=f1_bounds, inertia_weight=weight)
        best_position, best_score = pso.train()
        scores.append(best_score)
    
    average_score = np.mean(scores)
    results.append((weight, average_score))

for result in results:
    print(f"Inertia Weight: {result[0]}, Average Best Score: {result[1]}")

"""

for p in particles:
    scores = []
    for _ in range(10):  # Perform 10 iterations for each weight
        pso = PSO(num_particles=p, num_iterations=10000, obj_func=f1, bounds=f1_bounds, inertia_weight=0.7)
        best_position, best_score = pso.train()
        scores.append(best_score)
    
    average_score = np.mean(scores)
    results.append((p, average_score))

for result in results:
    print(f"Particles: {result[0]}, Average Best Score: {result[1]}")
