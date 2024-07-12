from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Definición de la estructura de la red
modelo_red_bayesiana = BayesianNetwork([
    ('FalloSistema', 'UsoAltoCPU'), 
    ('FalloSistema', 'AltaTemperatura'), 
    ('FalloSistema', 'ErroresMemoria'), 
    ('FalloSistema', 'FallosRed')
])

# Definición de las CPDs (Tablas de Probabilidades Condicionales)
cpd_fallo_sistema = TabularCPD(variable='FalloSistema', variable_card=2, values=[[0.7], [0.3]])
cpd_uso_cpu = TabularCPD(variable='UsoAltoCPU', variable_card=2,
                         values=[[0.9, 0.4],
                                 [0.1, 0.6]],
                         evidence=['FalloSistema'], evidence_card=[2])
cpd_temp_alta = TabularCPD(variable='AltaTemperatura', variable_card=2,
                           values=[[0.85, 0.3],
                                   [0.15, 0.7]],
                           evidence=['FalloSistema'], evidence_card=[2])
cpd_errores_memoria = TabularCPD(variable='ErroresMemoria', variable_card=2,
                                values=[[0.8, 0.2],
                                        [0.2, 0.8]],
                                evidence=['FalloSistema'], evidence_card=[2])
cpd_fallos_red = TabularCPD(variable='FallosRed', variable_card=2,
                            values=[[0.75, 0.25],
                                    [0.25, 0.75]],
                            evidence=['FalloSistema'], evidence_card=[2])

# Añadir las CPDs al modelo
modelo_red_bayesiana.add_cpds(cpd_fallo_sistema, cpd_uso_cpu, cpd_temp_alta, cpd_errores_memoria, cpd_fallos_red)

# Verificar si el modelo es válido
assert modelo_red_bayesiana.check_model()

# Crear un objeto de inferencia
inferencia = VariableElimination(modelo_red_bayesiana)

# Función para clasificar los componentes según umbrales
def clasificar_componente(valor, umbral_bajo, umbral_alto):
    return 0 if valor <= umbral_bajo else 1

# Base de conocimientos (5 computadoras, 1 con fallos)
computadoras = [
    {
        'id': 'PC1',
        'UsoAltoCPU': 75,       # 75% uso de CPU (fallando)
        'AltaTemperatura': 65,  # 65°C (fallando)
        'ErroresMemoria': 7,    # 7 errores por hora (fallando)
        'FallosRed': 5          # 5 fallos por hora (fallando)
    },
    {
        'id': 'PC2',
        'UsoAltoCPU': 45,       # 45% uso de CPU (bien)
        'AltaTemperatura': 55,  # 55°C (bien)
        'ErroresMemoria': 2,    # 2 errores por hora (bien)
        'FallosRed': 1          # 1 fallo por hora (bien)
    },
    {
        'id': 'PC3',
        'UsoAltoCPU': 50,       # 50% uso de CPU (bien)
        'AltaTemperatura': 50,  # 50°C (bien)
        'ErroresMemoria': 3,    # 3 errores por hora (bien)
        'FallosRed': 2          # 2 fallos por hora (bien)
    },
    {
        'id': 'PC4',
        'UsoAltoCPU': 40,       # 40% uso de CPU (bien)
        'AltaTemperatura': 52,  # 52°C (bien)
        'ErroresMemoria': 1,    # 1 error por hora (bien)
        'FallosRed': 0          # 0 fallos por hora (bien)
    },
    {
        'id': 'PC5',
        'UsoAltoCPU': 55,       # 55% uso de CPU (bien)
        'AltaTemperatura': 54,  # 54°C (bien)
        'ErroresMemoria': 4,    # 4 errores por hora (bien)
        'FallosRed': 1          # 1 fallo por hora (bien)
    }
]

# Inicializar contadores para estadísticas
estadisticas_fallos = {
    'UsoAltoCPU': {'si': 0, 'no': 0},
    'AltaTemperatura': {'si': 0, 'no': 0},
    'ErroresMemoria': {'si': 0, 'no': 0},
    'FallosRed': {'si': 0, 'no': 0}
}
total_computadoras = len(computadoras)

# Clasificar y realizar la inferencia para cada computadora
for pc in computadoras:
    evidencia = {
        'UsoAltoCPU': clasificar_componente(pc['UsoAltoCPU'], 70, 100),
        'AltaTemperatura': clasificar_componente(pc['AltaTemperatura'], 60, 100),
        'ErroresMemoria': clasificar_componente(pc['ErroresMemoria'], 5, 10),
        'FallosRed': clasificar_componente(pc['FallosRed'], 3, 10)
    }
    
    # Actualizar estadísticas de fallos
    for componente, valor in evidencia.items():
        if valor == 1:
            estadisticas_fallos[componente]['si'] += 1
        else:
            estadisticas_fallos[componente]['no'] += 1

    try:
        prob_fallo_sistema = inferencia.query(variables=['FalloSistema'], evidence=evidencia)
        fallo_sistema = 'si' if prob_fallo_sistema.values[1] > 0.5 else 'no'
        print(f"Probabilidad de fallo en {pc['id']}:")
        print(prob_fallo_sistema)
        print(f"Diagnóstico de fallo del sistema para {pc['id']}: {fallo_sistema}")
    except Exception as e:
        print(f"Error al inferir para {pc['id']}: {e}")

# Calcular porcentajes de fallos
for componente, datos in estadisticas_fallos.items():
    datos['si'] = (datos['si'] / total_computadoras) * 100
    datos['no'] = (datos['no'] / total_computadoras) * 100

print("\nEstadísticas de fallos por componente:")
for componente, datos in estadisticas_fallos.items():
    print(f"{componente}: {datos}")

# Crear un gráfico de la red bayesiana
G = nx.DiGraph()
G.add_edges_from([
    ('FalloSistema', 'UsoAltoCPU'), 
    ('FalloSistema', 'AltaTemperatura'), 
    ('FalloSistema', 'ErroresMemoria'), 
    ('FalloSistema', 'FallosRed')
])
pos = nx.spring_layout(G)

plt.figure()
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_color='black')
plt.title('Red Bayesiana - Predicción de Fallo del Sistema')
plt.show()
