import random
import numpy as np

def generate_rotation_vector():
    # možné hodnoty pro x a y:
    possible_values = [0, 5, 10, -5, -10]

    while True:
        # náhodný výběr x:
        x = random.choice(possible_values)

        # podmíněný výběr y:
        if abs(x) == 10:
            y = random.choice([0, 5, -5])  # Omezení pokud je x = +-10
        else:
            y = random.choice(possible_values)

        # Opačné omezení pro x pokud y je +-10
        if abs(y) == 10 and abs(x) > 5:
            x = random.choice([0, 5, -5])

        # Náhodný výběr pro z (krok 45, max 315)
        z = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

        # Zabránění generování (0, 0, 0)
        if (x, y, z) != (0, 0, 0):
            return (x, y, z)

def random_vector_list(num_of_vectors):
    random_rotations = set()
    while len(random_rotations) < num_of_vectors:
        vector = generate_rotation_vector()
        random_rotations.add(vector)

    return list(random_rotations)