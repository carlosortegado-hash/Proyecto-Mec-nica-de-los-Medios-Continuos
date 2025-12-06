import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Configuración de símbolos globales
x, y, z, t = sp.symbols('x y z t')

def obtener_input_usuario():
    """Solicita al usuario las componentes de la velocidad."""
    print("\n" + "="*50)
    print("   SIMULADOR DE FLUIDOS (Lagrangiano vs Euleriano)")
    print("="*50)
    print("Instrucciones: Usa sintaxis Python (ej: 2*x, sin(t), x**2, exp(y))")
    
    u_str = input(">> Introduce Velocidad en X (u): ")
    v_str = input(">> Introduce Velocidad en Y (v): ")
    w_str = input(">> Introduce Velocidad en Z (w): ")
    
    try:
        # Convertir strings a expresiones simbólicas
        u = sp.sympify(u_str)
        v = sp.sympify(v_str)
        w = sp.sympify(w_str)
        return u, v, w
    except Exception as e:
        print(f"ERROR: No se pudieron interpretar las ecuaciones. {e}")
        return None

def analizar_campo(u, v, w):
    """Calcula propiedades vectoriales del campo."""
    V = [u, v, w]
    print("\n" + "-"*30)
    print("       RESULTADOS TEÓRICOS")
    print("-"*30)
    
    # 1. Estacionario o No Estacionario
    # Si dV/dt != 0 en alguna componente, es no estacionario
    es_estacionario = True
    for comp in V:
        if sp.diff(comp, t) != 0:
            es_estacionario = False
            break
            
    if es_estacionario:
        print("[TIPO] Campo ESTACIONARIO (Independiente del tiempo).")
    else:
        print("[TIPO] Campo NO ESTACIONARIO (Depende del tiempo).")

    # 2. Dimensionalidad
    vars_presentes = set()
    for comp in V:
        vars_presentes.update(comp.free_symbols)
    
    # Filtramos solo las variables espaciales x, y, z
    coords = {x, y, z}
    dims_activas = coords.intersection(vars_presentes)
    
    # Lógica simple para determinar dimensionalidad visual
    if w == 0 and z not in vars_presentes:
        print("[DIMENSIÓN] Flujo BIDIMENSIONAL (Planario 2D).")
    elif len(dims_activas) <= 1 and w == 0 and v == 0: 
         print(f"[DIMENSIÓN] Flujo UNIDIMENSIONAL.")
    else:
        print("[DIMENSIÓN] Flujo TRIDIMENSIONAL (3D).")

    # 3. Divergencia (∇·V) -> Conservación de masa
    div_V = sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z)
    div_V = sp.simplify(div_V)
    print(f"\n[DIVERGENCIA ∇·V] = {div_V}")
    if div_V == 0:
        print("   -> Fluido INCOMPRESIBLE (Densidad constante, tipo líquido).")
    else:
        print("   -> Fluido COMPRESIBLE (Densidad variable, tipo gas).")

    # 4. Rotacional (∇xV) -> Vorticidad
    rot_x = sp.diff(w, y) - sp.diff(v, z)
    rot_y = sp.diff(u, z) - sp.diff(w, x)
    rot_z = sp.diff(v, x) - sp.diff(u, y)
    
    rotacional = [sp.simplify(rot_x), sp.simplify(rot_y), sp.simplify(rot_z)]
    print(f"\n[ROTACIONAL ∇xV] = ({rotacional[0]})i + ({rotacional[1]})j + ({rotacional[2]})k")
    
    if all(comp == 0 for comp in rotacional):
        print("   -> Flujo IRROTACIONAL (Potencial).")
    else:
        print("   -> Flujo ROTACIONAL (Tiene vorticidad).")
        
    return es_estacionario

def simular(u_expr, v_expr, w_expr, tipo_linea):
    """Ejecuta la animación visual con Matplotlib."""
    print(f"\nGenerando simulación: {tipo_linea.upper()}...")
    print("Cierra la ventana del gráfico para terminar el programa.")

    # Convertir expresiones simbólicas a funciones numéricas rápidas (lambda)
    # Deben aceptar argumentos (t, x, y, z) aunque no los usen todos
    u_func = sp.lambdify((t, x, y, z), u_expr, 'numpy')
    v_func = sp.lambdify((t, x, y, z), v_expr, 'numpy')
    w_func = sp.lambdify((t, x, y, z), w_expr, 'numpy')

    # Configuración de Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Límites del cubo de visualización
    L = 5
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    
    title = ax.set_title(f"Simulación: {tipo_linea}")

    # Función auxiliar para obtener vector velocidad numérico
    def get_velocity(time_val, pos):
        px, py, pz = pos
        # Manejo de excepciones por si el lambdify devuelve un escalar puro
        try: vx = float(u_func(time_val, px, py, pz))
        except: vx = 0.0
        try: vy = float(v_func(time_val, px, py, pz))
        except: vy = 0.0
        try: vz = float(w_func(time_val, px, py, pz))
        except: vz = 0.0
        return np.array([vx, vy, vz])

    # ---------------------------------------------------------
    # OPCIÓN 1: TRAYECTORIA (Pathline) - Lagrangiano
    # ---------------------------------------------------------
    if tipo_linea == 'trayectoria':
        print(">> Visualizando el camino histórico de partículas individuales.")
        
        # Puntos iniciales
        starts = [[0,0,0], [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,1]]
        lines = [ax.plot([], [], [], 'o-', markersize=3, label=f'Partícula {i}')[0] for i in range(len(starts))]
        paths = [[np.array(p)] for p in starts] # Historial
        
        def update_trayectoria(frame):
            dt = 0.05
            current_time = frame * dt
            
            for i, line in enumerate(lines):
                curr_pos = paths[i][-1]
                # Método de Euler: r_new = r_old + V(r, t) * dt
                vel = get_velocity(current_time, curr_pos)
                new_pos = curr_pos + vel * dt
                
                # Guardar y pintar
                paths[i].append(new_pos)
                data = np.array(paths[i])
                line.set_data(data[:, 0], data[:, 1])
                line.set_3d_properties(data[:, 2])
            
            title.set_text(f"Trayectoria (Pathline) - t: {current_time:.2f}s")
            return lines

        ani = FuncAnimation(fig, update_trayectoria, frames=200, interval=30, blit=False)
        plt.legend()
        plt.show()

    # ---------------------------------------------------------
    # OPCIÓN 2: LÍNEAS DE CORRIENTE (Streamlines) - Euleriano
    # ---------------------------------------------------------
    elif tipo_linea == 'corriente':
        print(">> Visualizando el campo de vectores instantáneo (foto fija).")
        
        # Malla de puntos semilla
        seeds = []
        for sx in np.linspace(-2, 2, 3):
            for sy in np.linspace(-2, 2, 3):
                seeds.append([sx, sy, 0])
        
        lines = [ax.plot([], [], [], color='blue', alpha=0.6)[0] for _ in seeds]

        def update_corriente(frame):
            # El tiempo avanza lentamente para ver cómo evoluciona la "foto" del campo
            fixed_time = frame * 0.1 
            
            for i, start_pos in enumerate(seeds):
                # Función para odeint (dx/ds = u, dy/ds = v...)
                # El tiempo está FIJO en esta integración
                def dpos_ds(pos, s):
                    return get_velocity(fixed_time, pos)
                
                s_span = np.linspace(0, 4, 30) # Longitud de la línea
                path = odeint(dpos_ds, start_pos, s_span)
                
                lines[i].set_data(path[:, 0], path[:, 1])
                lines[i].set_3d_properties(path[:, 2])
                
            title.set_text(f"Líneas de Corriente (t fijo = {fixed_time:.2f})")
            return lines

        ani = FuncAnimation(fig, update_corriente, frames=100, interval=50, blit=False)
        plt.show()

    # ---------------------------------------------------------
    # OPCIÓN 3: LÍNEAS DE HUMO (Streaklines) - Experimental
    # ---------------------------------------------------------
    elif tipo_linea == 'humo':
        print(">> Visual
