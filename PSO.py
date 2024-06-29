import numpy as np

class PSO:
    def __init__(self, num_particles, num_iterations, obj_func, dim=20, bounds=(0,1), inertia_weight=0.7):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.obj_func = obj_func
        self.dim = dim 
        self.bounds = bounds
        self.inertia_weight = inertia_weight
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(num_particles, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def train(self, iteration_number):
        with open(f"result_{iteration_number}.txt", "w") as file:
            for iteration in range(self.num_iterations):
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
                    
                    self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                
                file.write(f"Iteracion {iteration+1}: Mejor valor = {self.global_best_score}\n")
        
        return self.global_best_position, self.global_best_score


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

particles = [100]
results = []

for p in particles:
    scores = []
    for i in range(10):
        pso = PSO(num_particles=p, num_iterations=5000, obj_func=f2, bounds=f2_bounds, inertia_weight=0.7)
        best_position, best_score = pso.train(iteration_number=i+1)
        scores.append(best_score)
    
    average_score = np.mean(scores)
    results.append((p, average_score))

for result in results:
    print(f"Particles: {result[0]}, Average Best Score: {result[1]}")

# Leer los archivos generados y calcular el promedio de cada iteración
num_iterations = 5000
num_executions = 10
iteration_scores = np.zeros((num_executions, num_iterations))

for i in range(num_executions):
    with open(f"result_{i+1}.txt", "r") as file:
        for j, line in enumerate(file):
            score = float(line.split('=')[1].strip())
            iteration_scores[i, j] = score

average_iteration_scores = np.mean(iteration_scores, axis=0)

with open("average_results.txt", "w") as file:
    for iteration in range(num_iterations):
        file.write(f"Iteracion {iteration+1}: Promedio Mejor valor = {average_iteration_scores[iteration]}\n")

print("Promedios de cada iteración guardados en 'average_results.txt'")
