import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Definimos las variables matemáticas
x, y, z, t = sp.symbols('x y z t')

def ejecutar_programa():
    print("--- SIMULADOR DE FLUIDOS ---")
    print("Ejemplos de input: 2*x, -y, 0.1*t, sin(x)")
    
    # 1. PEDIR DATOS
    try:
        u_str = input("Velocidad en X (u): ")
        v_str = input("Velocidad en Y (v): ")
        w_str = input("Velocidad en Z (w): ")
        
        # Convertir texto a matemáticas
        u = sp.sympify(u_str)
        v = sp.sympify(v_str)
        w = sp.sympify(w_str)
    except:
        print("Error: Escribe bien las fórmulas.")
        return

    # 2. CALCULAR PROPIEDADES (Divergencia y Rotacional)
    print("\n--- RESULTADOS ---")
    
    # Estacionario (si depende del tiempo t)
    if sp.diff(u, t) != 0 or sp.diff(v, t) != 0 or sp.diff(w, t) != 0:
        print("TIPO: No Estacionario (depende del tiempo).")
    else:
        print("TIPO: Estacionario (constante).")

    # Divergencia (si es líquido o gas)
    div = sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z)
    print(f"DIVERGENCIA: {sp.simplify(div)}")
    if div == 0: print("(Incompresible / Líquido)")
    else: print("(Compresible / Gas)")

    # Rotacional (si gira)
    rot_x = sp.diff(w, y) - sp.diff(v, z)
    rot_y = sp.diff(u, z) - sp.diff(w, x)
    rot_z = sp.diff(v, x) - sp.diff(u, y)
    print(f"ROTACIONAL: ({sp.simplify(rot_x)})i + ({sp.simplify(rot_y)})j + ({sp.simplify(rot_z)})k")

    # 3. VISUALIZACIÓN
    print("\n¿Qué quieres ver?")
    print("1. Lineas de Corriente (Streamlines)")
    print("2. Trayectorias (Pathlines)")
    opcion = input("Elige 1 o 2: ")

    # Preparamos la gráfica
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # Convertimos formulas a funciones numéricas
    f_u = sp.lambdify((t, x, y, z), u, 'numpy')
    f_v = sp.lambdify((t, x, y, z), v, 'numpy')
    f_w = sp.lambdify((t, x, y, z), w, 'numpy')

    def obtener_vel(tiempo, pos):
        # Calcula velocidad en un punto y tiempo concreto
        px, py, pz = pos
        try: vx = float(f_u(tiempo, px, py, pz))
        except: vx = 0.0
        try: vy = float(f_v(tiempo, px, py, pz))
        except: vy = 0.0
        try: vz = float(f_w(tiempo, px, py, pz))
        except: vz = 0.0
        return np.array([vx, vy, vz])

    if opcion == '1': # LINEAS DE CORRIENTE
        print("Generando líneas de corriente...")
        # Puntos de inicio
        semillas = [[i, j, 0] for i in range(-2, 3, 2) for j in range(-2, 3, 2)]
        for p in semillas:
            # Dibujamos una linea para cada punto
            # En corriente el tiempo está CONGELADO (t=0 para simplificar)
            t_fijo = 0 
            camino = [p]
            curr = np.array(p, dtype=float)
            for _ in range(50): # 50 pasos
                v = obtener_vel(t_fijo, curr)
                curr = curr + v * 0.1
                camino.append(curr)
            
            camino = np.array(camino)
            ax.plot(camino[:,0], camino[:,1], camino[:,2], color='blue')
            
    elif opcion == '2': # TRAYECTORIAS
        print("Generando trayectorias...")
        p = np.array([1.0, 1.0, 0.0]) # Una partícula
        linea, = ax.plot([], [], [], 'r-o')
        
        historial = [p]
        def update(frame):
            nonlocal p
            ti = frame * 0.1
            v = obtener_vel(ti, p)
            p = p + v * 0.1 # Movemos la particula
            historial.append(p)
            dat = np.array(historial)
            linea.set_data(dat[:,0], dat[:,1])
            linea.set_3d_properties(dat[:,2])
            return linea,

        anim = FuncAnimation(fig, update, frames=100, interval=50)

    plt.show()

# Ejecutar
if __name__ == "__main__":
    ejecutar_programa()
