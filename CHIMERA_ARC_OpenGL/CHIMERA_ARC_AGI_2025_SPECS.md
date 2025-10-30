# CHIMERA v10.0 - Especificaciones ARC-AGI 2025
## Adaptación Completa al Dataset y Formato de Competición

**Francisco Angulo de Lafuente - CHIMERA Project 2025**

---

## 📊 ESPECIFICACIONES EXACTAS DEL DATASET

### 1. PALETA DE COLORES (CRITICAL)

**10 colores discretos: 0-9**

```python
# Paleta oficial ARC-AGI
ARC_COLOR_PALETTE = {
    0: (0, 0, 0),         # Negro (BACKGROUND - el más importante)
    1: (0, 116, 217),     # Azul
    2: (255, 65, 54),     # Rojo
    3: (46, 204, 64),     # Verde
    4: (255, 220, 0),     # Amarillo
    5: (170, 170, 170),   # Gris
    6: (240, 18, 190),    # Magenta/Fucsia
    7: (255, 133, 27),    # Naranja
    8: (127, 219, 255),   # Celeste/Cyan
    9: (135, 12, 37)      # Marrón/Granate
}

# CRITICAL: 0 = BACKGROUND
# En la mayoría de tareas, 0 es el fondo y 1-9 son objetos/formas
```

**Implicaciones para CHIMERA:**
- El canal R ya normaliza correctamente: `color / 9.0` ✅
- Pero debes distinguir **background (0) vs objetos (1-9)**
- Muchas operaciones requieren "sin contar el background"

---

### 2. TAMAÑOS DE GRID (DYNAMIC SIZING)

**Rango: 1×1 hasta 30×30**

```python
# Distribución estadística aproximada (basada en análisis del dataset público):
GRID_SIZE_STATS = {
    'tiny': (1, 5),      # ~5% de tareas
    'small': (6, 10),    # ~25% de tareas
    'medium': (11, 20),  # ~45% de tareas (más común)
    'large': (21, 30)    # ~25% de tareas
}

# CRITICAL: Input y Output pueden tener DIFERENTES tamaños
# Ejemplos reales:
# - Input: 15×15 → Output: 3×3 (extracción)
# - Input: 3×5 → Output: 9×15 (expansión/tiling)
# - Input: 7×7 → Output: 7×7 (transformación in-place)
```

**Implicaciones para CHIMERA:**
- Tu código actual usa tamaño fijo en `NeuromorphicFrame` ✅
- Pero necesitas detectar MEJOR el tamaño de output
- El `predict_next_in_sequence` es un inicio, pero insuficiente

---

### 3. FORMATO DE DATOS (JSON)

```json
{
  "train": [
    {
      "input": [[0, 1, 2], [3, 4, 5]],
      "output": [[5, 4, 3], [2, 1, 0]]
    },
    {
      "input": [[1, 1, 0], [0, 2, 2]],
      "output": [[2, 2, 0], [0, 1, 1]]
    }
  ],
  "test": [
    {
      "input": [[0, 3, 3], [4, 0, 4]]
    }
  ]
}
```

**Características:**
- Típicamente **2-5 ejemplos de entrenamiento** por tarea
- Típicamente **1-2 casos de prueba** por tarea
- Cada tarea es **única** (no hay dos tareas iguales)
- **Pixel-perfect**: Debe coincidir 100% (todos los píxeles correctos)

---

## 🎯 TIPOS DE TRANSFORMACIONES ESPACIALES

### Categorías Principales (Basado en análisis de Hodel's DSL - 165 operadores)

#### **A. TRANSFORMACIONES GEOMÉTRICAS** (~25% de tareas)

```python
GEOMETRIC_TRANSFORMS = {
    # Rotaciones
    'rotate_90': "Rotar 90° en sentido horario",
    'rotate_180': "Rotar 180°",
    'rotate_270': "Rotar 270° (= -90°)",
    
    # Reflexiones
    'reflect_horizontal': "Espejo horizontal (eje vertical)",
    'reflect_vertical': "Espejo vertical (eje horizontal)",
    'reflect_diagonal': "Espejo diagonal ↘",
    'reflect_antidiagonal': "Espejo antidiagonal ↙",
    
    # Transposiciones
    'transpose': "Transponer matriz (filas ↔ columnas)",
    
    # Combinaciones
    'rotate_then_reflect': "Composición de operaciones"
}
```

**Ejemplo típico:**
```
Input:        Output (rotate_90 + reflect_h):
[1, 2, 3]     [7, 4, 1]
[4, 5, 6]  →  [8, 5, 2]
[7, 8, 9]     [9, 6, 3]
```

#### **B. OPERACIONES SOBRE OBJETOS** (~35% de tareas)

```python
OBJECT_OPERATIONS = {
    # Extracción
    'extract_objects': "Detectar componentes conectados",
    'filter_by_size': "Filtrar objetos por área",
    'filter_by_color': "Filtrar por color",
    'largest_object': "Extraer el objeto más grande",
    
    # Transformación
    'move_object': "Mover objeto a nueva posición",
    'copy_object': "Duplicar objeto",
    'scale_object': "Escalar objeto (agrandar/reducir)",
    
    # Composición
    'overlay': "Superponer objetos",
    'mosaic': "Crear mosaico/patrón repetido",
    'fill_holes': "Rellenar huecos dentro de objetos",
    
    # Análisis
    'count_objects': "Contar número de objetos",
    'bounding_box': "Calcular caja delimitadora",
    'centroid': "Calcular centro de masa"
}
```

**Definición de "Objeto Conectado":**
- Células **contiguas** del **mismo color** (≠ 0)
- Conectividad: **4-vecinos** (arriba, abajo, izq, der) o **8-vecinos** (incluye diagonales)
- Mayormente se usa **4-conectividad**

#### **C. OPERACIONES DE COLOR** (~20% de tareas)

```python
COLOR_OPERATIONS = {
    # Mapeo directo
    'recolor': "Cambiar color X → Y",
    'swap_colors': "Intercambiar dos colores",
    
    # Condicional
    'paint_if': "Pintar según condición (ej: si tiene vecino rojo)",
    'floodfill': "Rellenar región conectada",
    
    # Patrón
    'checkerboard': "Patrón tablero de ajedrez",
    'stripe_pattern': "Patrón de rayas"
}
```

#### **D. OPERACIONES DE REJILLA/GRID** (~15% de tareas)

```python
GRID_OPERATIONS = {
    # Estructura
    'detect_grid': "Detectar estructura de rejilla/mosaico",
    'tile_repeat': "Repetir patrón en rejilla N×M",
    'extract_cell': "Extraer celda de rejilla",
    
    # Simetría
    'detect_symmetry': "Detectar eje de simetría",
    'make_symmetric': "Hacer simétrico respecto a eje",
    
    # Bordes
    'expand_border': "Expandir borde por k píxeles",
    'crop_background': "Recortar background excesivo"
}
```

#### **E. LÓGICA COMPOSITIONAL** (~15% de tareas - ARC-AGI-2 emphasis)

```python
COMPOSITIONAL_LOGIC = {
    # Multi-regla
    'if_then_else': "Si condición → regla A, sino → regla B",
    'context_dependent': "Aplicar regla según contexto local",
    
    # Secuencial
    'step1_then_step2': "Aplicar transformación 1, luego 2",
    'recursive': "Aplicar regla recursivamente hasta convergencia",
    
    # Abstracción
    'define_symbol': "Definir nuevo símbolo/concepto en contexto",
    'substitute': "Sustituir patrón A por patrón B"
}
```

---

## 🧠 CORE KNOWLEDGE PRIORS (Chollet's Framework)

### Priors Fundamentales que ARC Asume:

```python
CORE_PRIORS = {
    # 1. Objectness (Permanencia de objetos)
    'object_persistence': "Objetos mantienen identidad al moverse",
    'object_cohesion': "Objetos son unidades coherentes",
    
    # 2. Goal-directedness
    'intentionality': "Transformaciones tienen propósito",
    
    # 3. Numbers & Counting
    'basic_arithmetic': "Contar, sumar, restar, multiplicar",
    'cardinality': "Concepto de 'cuántos'",
    
    # 4. Geometry & Topology
    'containment': "Dentro/fuera, adentro/afuera",
    'connectivity': "Conexión, adyacencia",
    'symmetry': "Simetría axial y rotacional",
    'basic_shapes': "Línea, rectángulo, cruz, L-shape",
    
    # 5. Agent-centric priors
    'contact': "Contacto físico, tocar",
    'support': "Objetos 'caen' por gravedad simulada"
}
```

---

## 🚀 ADAPTACIONES CRÍTICAS PARA CHIMERA v10.0

### 1. **NORMALIZACIÓN DE COLORES (REQUERIDO)**

```python
# ACTUAL (v9.5):
data[:h, :w, 0] = grid.astype(float) / 9.0  # R = estado

# PROBLEMA: 0 y 1 están muy cerca (0.0 vs 0.111)
# SOLUCIÓN v10.0:

def normalize_color(color: int) -> float:
    """
    Normalización especial que separa background de objetos.
    """
    if color == 0:
        return 0.0  # Background
    else:
        return (color / 9.0) * 0.9 + 0.1  # Objetos: [0.1, 1.0]
    
# Esto da:
# 0 → 0.0
# 1 → 0.2
# 2 → 0.3
# ...
# 9 → 1.0
```

### 2. **DETECCIÓN DE TAMAÑO OUTPUT (MEJORADO)**

```python
def predict_output_size_v10(self, train_examples, test_input_shape):
    """
    Heurísticas mejoradas para predecir tamaño de output.
    """
    in_sizes = [np.array(ex['input']).shape for ex in train_examples]
    out_sizes = [np.array(ex['output']).shape for ex in train_examples]
    
    # 1. Detectar patrones conocidos
    
    # Identity: input == output
    if all(i == o for i, o in zip(in_sizes, out_sizes)):
        return test_input_shape
    
    # Constant: todos los outputs iguales
    if len(set(out_sizes)) == 1:
        return out_sizes[0]
    
    # Ratios constantes (scale)
    ratios_h = [o[0] / i[0] for i, o in zip(in_sizes, out_sizes)]
    ratios_w = [o[1] / i[1] for i, o in zip(in_sizes, out_sizes)]
    
    if len(set([round(r, 2) for r in ratios_h])) == 1:
        # Ratio consistente
        ratio_h = ratios_h[0]
        ratio_w = ratios_w[0]
        return (
            int(test_input_shape[0] * ratio_h),
            int(test_input_shape[1] * ratio_w)
        )
    
    # 2. Heurísticas específicas de ARC
    
    # Crop to content (eliminar background)
    # Ejemplo: 30×30 con objeto 5×5 → output 5×5
    
    # Grid extraction
    # Ejemplo: rejilla 3×3 de celdas 2×2 → output podría ser 3×3
    
    # Default: mantener tamaño input
    return test_input_shape
```

### 3. **OPERADOR 3×3 CON BACKGROUND-AWARE**

```glsl
// v10.0: Operador espacial consciente de background
uniform sampler2D u_state;
uniform int u_color_map[10];
uniform ivec2 grid_size;

out vec4 out_frame;

void main() {
    ivec2 coord = ivec2(uv * grid_size);
    coord = clamp(coord, ivec2(0), grid_size - ivec2(1));
    
    // Leer estado actual
    vec4 center = texelFetch(u_state, coord, 0);
    int center_color = int(center.r * 9.0 + 0.5);
    
    // Análisis de vecindario 3×3
    int neighbor_colors[8];
    int idx = 0;
    for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
            if(dx == 0 && dy == 0) continue; // skip center
            
            ivec2 ncoord = coord + ivec2(dx, dy);
            ncoord = clamp(ncoord, ivec2(0), grid_size - ivec2(1));
            
            vec4 neighbor = texelFetch(u_state, ncoord, 0);
            neighbor_colors[idx++] = int(neighbor.r * 9.0 + 0.5);
        }
    }
    
    // Features locales (IGNORING BACKGROUND)
    int same_color_count = 0;      // Vecinos del mismo color
    int non_bg_neighbors = 0;      // Vecinos que NO son background
    bool is_edge = false;          // ¿Está en borde de objeto?
    bool touches_bg = false;       // ¿Toca el background?
    
    for(int i = 0; i < 8; i++) {
        if(neighbor_colors[i] != 0) {
            non_bg_neighbors++;
        }
        
        if(neighbor_colors[i] == center_color) {
            same_color_count++;
        }
        
        if(neighbor_colors[i] == 0 && center_color != 0) {
            touches_bg = true;
        }
    }
    
    // Es borde si: tiene vecinos diferentes O toca background
    is_edge = (same_color_count < non_bg_neighbors) || touches_bg;
    
    // Emitir features a canal G (memoria)
    float edge_strength = is_edge ? 1.0 : 0.0;
    float density = float(same_color_count) / 8.0;
    
    // Aplicar transformación de color
    int output_color = u_color_map[center_color];
    float result_val = float(output_color) / 9.0;
    
    out_frame = vec4(
        center.r,           // R: estado original
        edge_strength,      // G: ¿es borde?
        result_val,         // B: resultado transformado
        density             // A: densidad local
    );
}
```

### 4. **DSL MÍNIMA ADAPTADA A ARC**

```python
class CHIMERA_ARC_DSL:
    """
    Operadores GPU para ARC-AGI, compilados como shaders.
    """
    
    # TIER 1: Geométricos (más frecuentes - 25%)
    @staticmethod
    def rotate_90(frame): pass
    
    @staticmethod
    def rotate_180(frame): pass
    
    @staticmethod
    def reflect_horizontal(frame): pass
    
    @staticmethod
    def reflect_vertical(frame): pass
    
    @staticmethod
    def transpose(frame): pass
    
    # TIER 2: Objetos (35%)
    @staticmethod
    def extract_largest_object(frame): pass
    
    @staticmethod
    def fill_holes(frame): pass
    
    @staticmethod
    def copy_to_position(frame, obj_id, x, y): pass
    
    @staticmethod
    def scale_object(frame, obj_id, factor): pass
    
    @staticmethod
    def tile_pattern(frame, n, m): pass
    
    # TIER 3: Color (20%)
    @staticmethod
    def recolor(frame, old_c, new_c): pass
    
    @staticmethod
    def floodfill(frame, x, y, color): pass
    
    # TIER 4: Grid (15%)
    @staticmethod
    def expand_border(frame, k): pass
    
    @staticmethod
    def crop_to_content(frame): pass
    
    @staticmethod
    def detect_and_tile(frame): pass
    
    # TIER 5: Compositional (5% pero CRITICAL en ARC-AGI-2)
    @staticmethod
    def if_color_then_transform(frame, color, transform): pass
```

---

## 📈 ESTRATEGIA DE SCORING EN KAGGLE

### Formato de Submission

```python
# Debes devolver 2 intentos por cada test case:
submission = [
    {
        'task_id_1': [
            [[attempt_1_grid]],  # Intento 1
            [[attempt_2_grid]]   # Intento 2
        ]
    },
    {
        'task_id_2': [
            [[attempt_1_grid]],
            [[attempt_2_grid]]
        ]
    }
]

# Score = (# de tareas con al menos 1 intento correcto) / (# total de tareas)
# Si intento_1 O intento_2 coincide 100% → score = 1 para esa tarea
# Si ninguno coincide → score = 0
```

### Estrategia Dual Attempt para CHIMERA

```python
def generate_dual_attempts(task, pattern_confidence):
    """
    Estrategia de 2 intentos optimizada.
    """
    
    # Attempt 1: Usar patrón decodificado con mayor confianza
    attempt_1 = apply_decoded_pattern(task)
    
    if pattern_confidence > 0.7:
        # Alta confianza: segundo intento con pequeña variación
        attempt_2 = apply_pattern_with_augmentation(task)
    elif pattern_confidence > 0.4:
        # Confianza media: intentar transformación geométrica alternativa
        attempt_2 = try_geometric_variant(task)
    else:
        # Baja confianza: beam search con DSL
        attempt_2 = beam_search_dsl(task, beam_width=4, max_depth=2)
    
    return [attempt_1, attempt_2]
```

---

## 🎯 MÉTRICAS DE ÉXITO

### Objetivo Kaggle 2025

```python
TARGET_METRICS = {
    'grand_prize': {
        'accuracy': 0.85,           # 85% en private eval (120 tareas)
        'efficiency': '$2.5/task',  # ~102 tareas con $250 presupuesto
        'time_limit': '12 hours'    # Wall-clock time
    },
    
    'competitive_score': {
        'accuracy': 0.55,           # 55% (nivel MindsAI 2024)
        'efficiency': 'dentro de límites L4×4'
    },
    
    'realistic_v10': {
        'accuracy': 0.30,           # 30% con Ola 1 (spatial awareness)
        'accuracy_ola2': 0.45,      # 45% con Ola 2 (object-level)
        'accuracy_ola3': 0.60       # 60% con Ola 3 (DSL + beam search)
    }
}
```

### Recursos Disponibles

```python
KAGGLE_LIMITS = {
    'gpu': 'NVIDIA L4 × 4',
    'vram': '96 GB total (24 GB por L4)',
    'time': '12 hours wall-clock',
    'internet': False,  # NO internet access!
    'storage': 'limitado a notebook'
}
```

---

## 🔧 CHECKLIST DE IMPLEMENTACIÓN v10.0

### Sprint 1: Foundation (Semana 1)
- [ ] Actualizar normalización de colores con background-aware
- [ ] Mejorar `predict_output_size` con heurísticas ARC
- [ ] Implementar operador 3×3 con features locales
- [ ] Añadir positional encoding texture
- [ ] Test en 50 tareas de training set

### Sprint 2: Object-Level (Semana 2)
- [ ] Implementar Jump Flooding para componentes conectados
- [ ] Agregar features: tamaño, bbox, centroid por objeto
- [ ] Implementar mipmaps multi-escala
- [ ] Test en tareas de object manipulation

### Sprint 3: DSL Core (Semana 3-4)
- [ ] Implementar 15 operadores DSL en GPU (Tiers 1-3)
- [ ] Beam search básico (width=4, depth=2)
- [ ] Hungarian algorithm para color mapping
- [ ] Test en tareas complejas de training

### Sprint 4: Refinement (Semana 5)
- [ ] Convergencia adaptativa
- [ ] Validadores de invariantes
- [ ] Dual attempt strategy
- [ ] Test en evaluation set (sin mirar respuestas!)

### Sprint 5: Integration (Semana 6)
- [ ] Optimización de memoria para Kaggle
- [ ] Checkpoint/resume para 12-hour limit
- [ ] Logging y debug tools
- [ ] Submission script

---

## 📚 REFERENCIAS CLAVE

1. **Dataset Oficial**: https://github.com/arcprize/ARC-AGI-2
2. **Competición Kaggle**: https://www.kaggle.com/competitions/arc-prize-2025
3. **Guía Oficial**: https://arcprize.org/guide
4. **Paper ARC-AGI-2**: https://arxiv.org/abs/2505.11831
5. **Hodel's DSL**: https://github.com/michaelhodel/arc-dsl
6. **Chollet's Original Paper**: "On the Measure of Intelligence" (2019)

---

## 💡 INSIGHTS FINALES

### Lo que CHIMERA hace bien:
✅ GPU-native architecture (ventaja de velocidad)
✅ Neuromorphic loop (estado + memoria + resultado)
✅ Pattern decoder temporal (buen punto de partida)

### Lo que NECESITA:
🔴 Spatial awareness (vecindarios, bordes)
🔴 Object-level reasoning (componentes, formas)
🔴 Compositional logic (DSL + search)
🔴 Background handling (0 es especial!)

### La Ventaja Única de CHIMERA:
💪 **GPU "piensa visualmente"**: ARC son transformaciones visuales
💪 **Parallelismo masivo**: 96GB VRAM = múltiples hipótesis simultáneas
💪 **Neuromorphic = Emergencia**: No programas rígidos, sino "render=compute"

---

**Next Steps**: Implementar Ola 1 (Spatial Awareness) y validar en training set.

¿Objetivo realista para v10.0?  
**30-35% accuracy** en semi-private eval → Top 50% del leaderboard 🎯
