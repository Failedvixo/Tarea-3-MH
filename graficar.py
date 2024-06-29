import matplotlib.pyplot as plt

# Nombre del archivo
filename = 'resultf2.txt'

# Listas para almacenar los datos
iterations = []
average_best_values = []

# Leer el archivo
with open(filename, 'r') as file:
    for line in file:
        # Dividir la línea por los dos puntos y extraer los valores
        parts = line.split(':')
        if len(parts) == 2:
            iteration_part = parts[0].strip()
            value_part = parts[1].strip()
            
            # Extraer los números
            iteration_number = int(iteration_part.split()[1])
            average_best_value = float(value_part.split('=')[1].strip())
            
            # Añadir a las listas
            iterations.append(iteration_number)
            average_best_values.append(average_best_value)

# Graficar los datos
plt.plot(iterations, average_best_values, marker='o')
plt.xlabel('Iteración')
plt.ylabel('Promedio Mejor Valor')
plt.title('Iteración vs Promedio Mejor Valor f2')
plt.grid(True)
plt.show()
