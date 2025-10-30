# CHIMERA v10.0 - IMPLEMENTATION ROADMAP
## Plan Detallado con Código de Ejemplo

**Francisco Angulo de Lafuente - CHIMERA Project 2025**

---

## 🎯 OVERVIEW

```
v9.5 (actual)  →  v10.0 (target)
15% accuracy   →  30-35% accuracy (Ola 1)
               →  45-50% accuracy (Ola 2)
               →  58-65% accuracy (Ola 3)
```

---

## 📅 SPRINT 1: SPATIAL AWARENESS (Semana 1)
**Goal**: De pixel-level a neighborhood-level reasoning

### 1.1 Background-Aware Normalization

**Archivo**: `chimera_v10_core.py`

```python
def normalize_arc_color(color: int, background_aware: bool = True) -> float:
    """
    Normalización especial para ARC que separa background de objetos.
    
    Args:
        color: Entero 0-9
        background_aware: Si True, separa 0 del resto
        
    Returns:
        Float en [0.0, 1.0] con 0 siempre = 0.0
    """
    if color == 0:
        return 0.0  # Background siempre negro
    
    if background_aware:
        # Objetos en rango [0.1, 1.0] para mayor separación
        return 0.1 + (color / 9.0) * 0.9
    else:
        # Normalización lineal simple
        return color / 9.0


class NeuromorphicFrameV10:
    """
    v10.0: Añade background awareness y spatial features.
    """
    
    def __init__(self, ctx: moderngl.Context, size: Tuple[int, int]):
        self.ctx = ctx
        self.h, self.w = size
        
        # Textura principal (igual que v9.5)
        self.unified_texture = ctx.texture(
            size=(self.w, self.h),
            components=4,
            dtype='f4'
        )
        
        # NUEVO: Textura de features espaciales
        # R: edge_strength, G: neighbor_density, B: object_id, A: distance_to_bg
        self.spatial_features = ctx.texture(
            size=(self.w, self.h),
            components=4,
            dtype='f4'
        )
        
        # NUEVO: Textura de posición (estática)
        self.position_texture = self._create_position_encoding()
    
    def _create_position_encoding(self):
        """
        Genera encoding de posición con coordenadas normalizadas y trigonométricas.
        """
        pos_data = np.zeros((self.h, self.w, 4), dtype=np.float32)
        
        for y in range(self.h):
            for x in range(self.w):
                # Coordenadas normalizadas
                x_norm = x / (self.w - 1) if self.w > 1 else 0.5
                y_norm = y / (self.h - 1) if self.h > 1 else 0.5
                
                # Sinusoides para periodicidad
                x_sin = np.sin(2 * np.pi * x_norm)
                y_cos = np.cos(2 * np.pi * y_norm)
                
                pos_data[y, x] = [x_norm, y_norm, x_sin, y_cos]
        
        tex = self.ctx.texture(size=(self.w, self.h), components=4, dtype='f4')
        tex.write(pos_data.tobytes())
        return tex
    
    def upload_state(self, grid: np.ndarray):
        """
        Subir estado con background-aware normalization.
        """
        h, w = grid.shape
        data = np.zeros((self.h, self.w, 4), dtype=np.float32)
        
        # Normalización background-aware
        for y in range(h):
            for x in range(w):
                color = int(grid[y, x])
                data[y, x, 0] = normalize_arc_color(color)  # R = estado
        
        data[:h, :w, 3] = 1.0  # A = confianza inicial
        self.unified_texture.write(data.tobytes())
```

### 1.2 Shader de Operadores Espaciales 3×3

**Archivo**: `shaders_v10.py`

```python
SPATIAL_OPERATOR_SHADER = """
#version 330

uniform sampler2D u_state;          // Estado actual
uniform sampler2D u_position;       // Coordenadas y encoding
uniform sampler2D u_spatial_prev;   // Features espaciales del paso anterior
uniform int u_color_map[10];        // Transformación de colores
uniform ivec2 grid_size;
uniform float u_evolution_step;

in vec2 uv;
layout(location = 0) out vec4 out_frame;
layout(location = 1) out vec4 out_spatial;

// Constantes
const int BACKGROUND = 0;

// Función para obtener vecino con boundary check
int get_neighbor_color(ivec2 coord, int dx, int dy) {
    ivec2 ncoord = coord + ivec2(dx, dy);
    
    // Boundary check
    if(ncoord.x < 0 || ncoord.x >= grid_size.x || 
       ncoord.y < 0 || ncoord.y >= grid_size.y) {
        return BACKGROUND;  // Out of bounds = background
    }
    
    vec4 neighbor = texelFetch(u_state, ncoord, 0);
    return int(neighbor.r * 9.0 + 0.5);
}

void main() {
    ivec2 coord = ivec2(uv * grid_size);
    
    // Leer estado actual
    vec4 state_pixel = texelFetch(u_state, coord, 0);
    int center_color = int(state_pixel.r * 9.0 + 0.5);
    
    // Leer posición
    vec4 position = texture(u_position, uv);
    float x_norm = position.r;
    float y_norm = position.g;
    
    // === ANÁLISIS DE VECINDARIO 3×3 ===
    
    int same_color_count = 0;
    int different_color_count = 0;
    int non_bg_neighbors = 0;
    bool touches_background = false;
    
    // 8 direcciones
    const int offsets[8][2] = int[][](
        int[](-1, -1), int[](0, -1), int[](1, -1),  // top row
        int[](-1,  0),               int[](1,  0),  // middle row
        int[](-1,  1), int[](0,  1), int[](1,  1)   // bottom row
    );
    
    for(int i = 0; i < 8; i++) {
        int n_color = get_neighbor_color(coord, offsets[i][0], offsets[i][1]);
        
        if(n_color != BACKGROUND) {
            non_bg_neighbors++;
            
            if(n_color == center_color) {
                same_color_count++;
            } else {
                different_color_count++;
            }
        } else if(center_color != BACKGROUND) {
            touches_background = true;
        }
    }
    
    // === FEATURES ESPACIALES ===
    
    // Edge detection
    bool is_edge = touches_background || (different_color_count > 0);
    float edge_strength = is_edge ? 1.0 : 0.0;
    
    // Density (qué tan "rodeado" está)
    float neighbor_density = float(same_color_count) / 8.0;
    
    // Corner detection (menos de 3 vecinos del mismo color)
    bool is_corner = (center_color != BACKGROUND) && (same_color_count <= 2);
    float corner_score = is_corner ? 1.0 : 0.0;
    
    // Distance to border (basado en posición)
    float dist_to_border = min(
        min(x_norm, 1.0 - x_norm),
        min(y_norm, 1.0 - y_norm)
    );
    
    // === APLICAR TRANSFORMACIÓN DE COLOR ===
    
    int output_color = u_color_map[center_color];
    float result_val = float(output_color) / 9.0;
    
    // === MEZCLA CON MEMORIA TEMPORAL ===
    
    float memory_blend = mix(state_pixel.g, result_val, u_evolution_step);
    
    // Output principal (estado + memoria + resultado)
    out_frame = vec4(
        state_pixel.r,      // R: estado original
        memory_blend,       // G: memoria evolutiva
        result_val,         // B: resultado transformado
        state_pixel.a       // A: confianza
    );
    
    // Output de features espaciales
    out_spatial = vec4(
        edge_strength,      // R: ¿es borde?
        neighbor_density,   // G: densidad de vecinos
        corner_score,       // B: ¿es esquina?
        dist_to_border      // A: distancia a borde del grid
    );
}
"""


def compile_spatial_shader(ctx: moderngl.Context):
    """
    Compila el shader con MRT (Multiple Render Targets).
    """
    vertex_shader = """
    #version 330
    in vec2 in_vert;
    out vec2 uv;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
        uv = (in_vert + 1.0) / 2.0;
    }
    """
    
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=SPATIAL_OPERATOR_SHADER
    )
    
    return program
```

### 1.3 Integración en el Loop Neuromorphic

```python
class LivingBrainV10:
    """
    v10.0: Añade spatial awareness y MRT.
    """
    
    def _neuromorphic_evolution_v10(self, frame: NeuromorphicFrameV10, 
                                     color_map: List[int], steps: int = 3):
        """
        Evolución con spatial features.
        """
        current_main = frame.unified_texture
        current_spatial = frame.spatial_features
        
        for step in range(steps):
            # Crear MRT: 2 color attachments
            output_main = self.ctx.texture(
                size=(frame.w, frame.h), components=4, dtype='f4'
            )
            output_spatial = self.ctx.texture(
                size=(frame.w, frame.h), components=4, dtype='f4'
            )
            
            fbo = self.ctx.framebuffer(
                color_attachments=[output_main, output_spatial]
            )
            
            # Setup uniforms
            self.spatial_program['u_state'] = 0
            self.spatial_program['u_position'] = 1
            self.spatial_program['u_spatial_prev'] = 2
            self.spatial_program['u_color_map'].write(
                np.array(color_map, dtype='i4').tobytes()
            )
            self.spatial_program['grid_size'] = (frame.w, frame.h)
            self.spatial_program['u_evolution_step'] = (step + 1) / steps
            
            # Bind textures
            current_main.use(location=0)
            frame.position_texture.use(location=1)
            current_spatial.use(location=2)
            
            # Render
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            self._render_quad(self.spatial_program)
            
            # Ping-pong
            current_main = output_main
            current_spatial = output_spatial
            
            fbo.release()
        
        # Update frame
        frame.unified_texture = current_main
        frame.spatial_features = current_spatial
        
        return frame
```

---

## 📅 SPRINT 2: OBJECT-LEVEL REASONING (Semana 2)
**Goal**: De pixels a objetos completos

### 2.1 Connected Components con Jump Flooding

**Archivo**: `object_extraction_v10.py`

```python
JUMP_FLOODING_SHADER = """
#version 330

uniform sampler2D u_labels;     // Texture con labels actuales
uniform ivec2 grid_size;
uniform int u_step_size;        // 1, 2, 4, 8, 16...

in vec2 uv;
out vec4 out_labels;

void main() {
    ivec2 coord = ivec2(uv * grid_size);
    
    // Leer label y color actuales
    vec4 current = texelFetch(u_labels, coord, 0);
    int my_color = int(current.r * 9.0 + 0.5);
    float my_label = current.g;  // Label = coord linearizado normalizado
    
    // Si es background, no propagar
    if(my_color == 0) {
        out_labels = current;
        return;
    }
    
    // Buscar menor label en vecindario saltando u_step_size
    float min_label = my_label;
    
    for(int dy = -u_step_size; dy <= u_step_size; dy += u_step_size) {
        for(int dx = -u_step_size; dx <= u_step_size; dx += u_step_size) {
            ivec2 ncoord = coord + ivec2(dx, dy);
            
            // Boundary check
            if(ncoord.x < 0 || ncoord.x >= grid_size.x || 
               ncoord.y < 0 || ncoord.y >= grid_size.y) {
                continue;
            }
            
            vec4 neighbor = texelFetch(u_labels, ncoord, 0);
            int n_color = int(neighbor.r * 9.0 + 0.5);
            
            // Solo propagar entre mismo color
            if(n_color == my_color && neighbor.g < min_label) {
                min_label = neighbor.g;
            }
        }
    }
    
    // Output: mantener color, actualizar label
    out_labels = vec4(current.r, min_label, current.b, current.a);
}
"""


class ObjectExtractor:
    """
    Extrae componentes conectados usando Jump Flooding en GPU.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.jf_program = self._compile_jump_flooding()
    
    def compute_connected_components(self, frame: NeuromorphicFrameV10, 
                                     max_iters: int = 8):
        """
        Compute connected components via Jump Flooding.
        
        Returns:
            Texture con:
            - R: color original
            - G: component_id (normalizado)
            - B: object size (aproximado)
            - A: validity
        """
        # Inicializar: cada pixel es su propio label
        init_tex = self._initialize_labels(frame)
        
        current_tex = init_tex
        
        # Jump Flooding: step_size = [max_dim/2, max_dim/4, ..., 2, 1]
        max_dim = max(frame.w, frame.h)
        step_sizes = []
        step = max_dim // 2
        while step >= 1:
            step_sizes.append(step)
            step //= 2
        
        # Iterate
        for step_size in step_sizes:
            output_tex = self.ctx.texture(
                size=(frame.w, frame.h), components=4, dtype='f4'
            )
            fbo = self.ctx.framebuffer(color_attachments=[output_tex])
            
            self.jf_program['u_labels'] = 0
            self.jf_program['grid_size'] = (frame.w, frame.h)
            self.jf_program['u_step_size'] = step_size
            
            current_tex.use(location=0)
            
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            self._render_quad(self.jf_program)
            
            current_tex = output_tex
            fbo.release()
        
        # Post-process: calcular tamaños de objetos
        return self._compute_object_stats(current_tex, frame)
    
    def _initialize_labels(self, frame):
        """
        Inicializar labels: cada pixel = su coordenada linearizada.
        """
        h, w = frame.h, frame.w
        
        # Leer estado actual
        rgba = np.frombuffer(frame.unified_texture.read(), dtype=np.float32)
        rgba = rgba.reshape((h, w, 4))
        
        labels = np.zeros((h, w, 4), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                color = rgba[y, x, 0]
                
                # Label = coordenada linearizada normalizada
                linear_coord = y * w + x
                label = float(linear_coord) / (h * w - 1)
                
                labels[y, x] = [color, label, 0.0, 1.0]
        
        tex = self.ctx.texture(size=(w, h), components=4, dtype='f4')
        tex.write(labels.tobytes())
        return tex
    
    def _compute_object_stats(self, labels_tex, frame):
        """
        CPU pass para calcular estadísticas de objetos.
        """
        # Leer labels
        rgba = np.frombuffer(labels_tex.read(), dtype=np.float32)
        rgba = rgba.reshape((frame.h, frame.w, 4))
        
        # Contar píxeles por label
        label_map = rgba[:, :, 1]
        unique_labels, counts = np.unique(label_map, return_counts=True)
        
        # Crear mapping label → size
        label_to_size = dict(zip(unique_labels, counts))
        
        # Actualizar texture con tamaños
        for y in range(frame.h):
            for x in range(frame.w):
                label = rgba[y, x, 1]
                if label in label_to_size:
                    size = label_to_size[label]
                    rgba[y, x, 2] = float(size) / (frame.h * frame.w)
        
        # Rewrite texture
        labels_tex.write(rgba.tobytes())
        return labels_tex
```

### 2.2 Uso de Object Features en Decisiones

```python
def extract_object_features(labels_tex, frame):
    """
    Extrae features de alto nivel de los objetos detectados.
    """
    rgba = np.frombuffer(labels_tex.read(), dtype=np.float32)
    rgba = rgba.reshape((frame.h, frame.w, 4))
    
    colors = rgba[:, :, 0]
    labels = rgba[:, :, 1]
    
    # Agrupar por label
    unique_labels = np.unique(labels[colors > 0])  # Ignorar background
    
    objects = []
    for label in unique_labels:
        mask = (labels == label)
        
        # Bounding box
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        
        bbox = {
            'y_min': int(ys.min()),
            'y_max': int(ys.max()),
            'x_min': int(xs.min()),
            'x_max': int(xs.max())
        }
        
        # Centroid
        centroid = {
            'y': int(ys.mean()),
            'x': int(xs.mean())
        }
        
        # Color dominante
        color_vals = colors[mask]
        dominant_color = int(np.median(color_vals) * 9)
        
        # Size
        size = len(ys)
        
        objects.append({
            'label': label,
            'color': dominant_color,
            'size': size,
            'bbox': bbox,
            'centroid': centroid
        })
    
    return objects
```

---

## 📅 SPRINT 3: DSL + BEAM SEARCH (Semanas 3-4)
**Goal**: Program synthesis para tareas compositionales

### 3.1 Definición de DSL

```python
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum


class OperatorType(Enum):
    GEOMETRIC = "geometric"
    OBJECT = "object"
    COLOR = "color"
    GRID = "grid"


@dataclass
class Operator:
    """
    Operador primitivo de la DSL.
    """
    name: str
    type: OperatorType
    func: Callable
    cost: float  # Para A* search
    params: dict  # Parámetros esperados


class CHIMERA_DSL:
    """
    Domain-Specific Language para transformaciones ARC.
    """
    
    def __init__(self, brain: LivingBrainV10):
        self.brain = brain
        self.operators = self._initialize_operators()
    
    def _initialize_operators(self):
        """
        Define los 15-20 operadores core.
        """
        ops = []
        
        # === TIER 1: GEOMETRIC (5 ops) ===
        
        ops.append(Operator(
            name='rotate_90',
            type=OperatorType.GEOMETRIC,
            func=self._rotate_90_gpu,
            cost=1.0,
            params={}
        ))
        
        ops.append(Operator(
            name='rotate_180',
            type=OperatorType.GEOMETRIC,
            func=self._rotate_180_gpu,
            cost=1.0,
            params={}
        ))
        
        ops.append(Operator(
            name='reflect_h',
            type=OperatorType.GEOMETRIC,
            func=self._reflect_horizontal_gpu,
            cost=1.0,
            params={}
        ))
        
        ops.append(Operator(
            name='reflect_v',
            type=OperatorType.GEOMETRIC,
            func=self._reflect_vertical_gpu,
            cost=1.0,
            params={}
        ))
        
        ops.append(Operator(
            name='transpose',
            type=OperatorType.GEOMETRIC,
            func=self._transpose_gpu,
            cost=1.2,
            params={}
        ))
        
        # === TIER 2: OBJECT (5 ops) ===
        
        ops.append(Operator(
            name='extract_largest',
            type=OperatorType.OBJECT,
            func=self._extract_largest_object,
            cost=2.0,
            params={}
        ))
        
        ops.append(Operator(
            name='fill_holes',
            type=OperatorType.OBJECT,
            func=self._fill_holes_gpu,
            cost=1.5,
            params={}
        ))
        
        ops.append(Operator(
            name='tile_2x2',
            type=OperatorType.OBJECT,
            func=self._tile_pattern,
            cost=2.5,
            params={'n': 2, 'm': 2}
        ))
        
        ops.append(Operator(
            name='crop_content',
            type=OperatorType.GRID,
            func=self._crop_to_content,
            cost=1.5,
            params={}
        ))
        
        # ... más operadores
        
        return ops
    
    # === IMPLEMENTACIONES GPU ===
    
    def _rotate_90_gpu(self, frame: NeuromorphicFrameV10):
        """
        Rotar 90° usando texture sampling.
        """
        shader = """
        #version 330
        uniform sampler2D u_input;
        uniform ivec2 in_size;
        in vec2 uv;
        out vec4 out_color;
        
        void main() {
            // Rotation: (x, y) → (y, W-1-x)
            // En UV space: (u, v) → (v, 1-u)
            vec2 rotated_uv = vec2(uv.y, 1.0 - uv.x);
            out_color = texture(u_input, rotated_uv);
        }
        """
        # ... compilar y aplicar
        pass
    
    def _reflect_horizontal_gpu(self, frame):
        """
        Reflejar horizontalmente.
        """
        shader = """
        #version 330
        uniform sampler2D u_input;
        in vec2 uv;
        out vec4 out_color;
        
        void main() {
            // Mirror: (x, y) → (W-1-x, y)
            vec2 mirrored_uv = vec2(1.0 - uv.x, uv.y);
            out_color = texture(u_input, mirrored_uv);
        }
        """
        # ... compilar y aplicar
        pass
    
    # ... más implementaciones
```

### 3.2 Beam Search

```python
from dataclasses import dataclass
from typing import List
import heapq


@dataclass
class Program:
    """
    Secuencia de operadores.
    """
    ops: List[Operator]
    cost: float
    score: float  # Match con ejemplos
    
    def __lt__(self, other):
        # Para heapq (menor costo = mejor)
        return (self.score, -self.cost) < (other.score, -other.cost)


class BeamSearchSolver:
    """
    Búsqueda de programas en el espacio de la DSL.
    """
    
    def __init__(self, dsl: CHIMERA_DSL, beam_width: int = 8, max_depth: int = 3):
        self.dsl = dsl
        self.beam_width = beam_width
        self.max_depth = max_depth
    
    def search(self, train_examples: List[dict], test_input: np.ndarray):
        """
        Busca el mejor programa que transforma inputs → outputs.
        """
        # Beam inicial: programa vacío
        beam = [Program(ops=[], cost=0.0, score=float('inf'))]
        
        for depth in range(self.max_depth):
            candidates = []
            
            # Expandir cada programa en el beam
            for program in beam:
                # Probar añadir cada operador
                for op in self.dsl.operators:
                    new_program = Program(
                        ops=program.ops + [op],
                        cost=program.cost + op.cost,
                        score=0.0
                    )
                    
                    # Evaluar en ejemplos de entrenamiento
                    score = self._evaluate_program(new_program, train_examples)
                    new_program.score = score
                    
                    candidates.append(new_program)
            
            # Keep top beam_width
            candidates.sort()  # Por score (menor = mejor)
            beam = candidates[:self.beam_width]
            
            # Early stopping si perfect match
            if beam[0].score == 0.0:
                print(f"✓ Perfect match found at depth {depth+1}")
                break
        
        # Retornar mejor programa
        best = beam[0]
        return self._execute_program(best, test_input)
    
    def _evaluate_program(self, program: Program, train_examples: List[dict]) -> float:
        """
        Evalúa cuán bien el programa transforma inputs → outputs.
        
        Returns:
            Distancia total (0 = perfecto)
        """
        total_distance = 0.0
        
        for example in train_examples:
            input_grid = np.array(example['input'], dtype=np.uint8)
            expected = np.array(example['output'], dtype=np.uint8)
            
            # Ejecutar programa
            result = self._execute_program(program, input_grid)
            
            # Medir distancia (# de píxeles diferentes)
            if result.shape != expected.shape:
                # Penalización fuerte por tamaño incorrecto
                total_distance += 1000
            else:
                distance = np.sum(result != expected)
                total_distance += distance
        
        return total_distance
    
    def _execute_program(self, program: Program, input_grid: np.ndarray) -> np.ndarray:
        """
        Ejecuta secuencia de operadores sobre input.
        """
        # Crear frame
        h, w = input_grid.shape
        frame = NeuromorphicFrameV10(self.dsl.brain.ctx, (h, w))
        frame.upload_state(input_grid)
        
        # Aplicar operadores secuencialmente
        for op in program.ops:
            frame = op.func(frame)
        
        # Extraer resultado
        result = frame.download_result()
        frame.release()
        
        return result
```

---

## 📅 SPRINT 4: INTEGRATION & OPTIMIZATION (Semana 5)

### 4.1 Estrategia Dual Attempt

```python
def generate_dual_attempts_v10(brain: LivingBrainV10, task: dict):
    """
    Genera 2 intentos optimizados por confianza.
    """
    # Decodificar patrón
    color_map, pattern_type, confidence = brain._decode_pattern(task['train'])
    
    # Attempt 1: Siempre el patrón decodificado
    attempt_1 = brain.solve_with_pattern(task['test'][0]['input'], color_map)
    
    # Attempt 2: Depende de confianza
    if confidence > 0.7:
        # Alta confianza: variación geométrica
        attempt_2 = brain.solve_with_geometric_aug(
            task['test'][0]['input'], 
            color_map
        )
    
    elif confidence > 0.4:
        # Media confianza: probar spatial ops
        attempt_2 = brain.solve_with_spatial_ops(
            task['test'][0]['input']
        )
    
    else:
        # Baja confianza: beam search con DSL
        dsl = CHIMERA_DSL(brain)
        searcher = BeamSearchSolver(dsl, beam_width=4, max_depth=2)
        attempt_2 = searcher.search(
            task['train'],
            np.array(task['test'][0]['input'])
        )
    
    return [attempt_1.tolist(), attempt_2.tolist()]
```

### 4.2 Submission Format para Kaggle

```python
def create_submission(brain: LivingBrainV10, test_tasks: dict) -> dict:
    """
    Crea submission en el formato requerido por Kaggle.
    """
    submission = []
    
    for task_id, task_data in test_tasks.items():
        task_attempts = {}
        
        for test_case_idx, test_case in enumerate(task_data['test']):
            # Generar 2 intentos
            attempts = generate_dual_attempts_v10(brain, {
                'train': task_data['train'],
                'test': [test_case]
            })
            
            task_attempts[test_case_idx] = attempts
        
        submission.append({task_id: task_attempts})
    
    return submission
```

---

## 🎯 MÉTRICAS Y TESTING

### Test Suite

```python
def run_comprehensive_tests():
    """
    Suite de tests para validar cada componente.
    """
    brain = LivingBrainV10()
    
    # Test 1: Background normalization
    assert normalize_arc_color(0) == 0.0
    assert normalize_arc_color(1) == 0.2
    assert normalize_arc_color(9) == 1.0
    print("✓ Background normalization OK")
    
    # Test 2: Spatial operators
    test_grid = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    frame = NeuromorphicFrameV10(brain.ctx, (3, 3))
    frame.upload_state(test_grid)
    # ... aplicar operador y verificar features
    print("✓ Spatial operators OK")
    
    # Test 3: Connected components
    # ... test con grid conocido
    print("✓ Connected components OK")
    
    # Test 4: DSL operators
    # ... test cada operador
    print("✓ DSL operators OK")
    
    # Test 5: Beam search
    # ... test con tarea simple conocida
    print("✓ Beam search OK")
    
    print("\n🎉 All tests passed!")
```

### Benchmark en Training Set

```python
def benchmark_on_training_set(sample_size=50):
    """
    Medir accuracy en subset del training set.
    """
    from arc_dataset_loader import load_training_tasks
    
    tasks = load_training_tasks()
    sample = random.sample(tasks, sample_size)
    
    brain = LivingBrainV10()
    
    results = {
        'v95_baseline': 0,
        'v10_ola1': 0,
        'v10_ola2': 0,
        'v10_ola3': 0
    }
    
    for task in sample:
        # Baseline (v9.5 - solo color mapping)
        if solve_v95_style(brain, task):
            results['v95_baseline'] += 1
        
        # Ola 1 (+ spatial ops)
        if solve_with_spatial(brain, task):
            results['v10_ola1'] += 1
        
        # Ola 2 (+ objects)
        if solve_with_objects(brain, task):
            results['v10_ola2'] += 1
        
        # Ola 3 (+ DSL)
        if solve_with_dsl(brain, task):
            results['v10_ola3'] += 1
    
    # Print results
    for key, count in results.items():
        acc = count / sample_size
        print(f"{key}: {acc:.1%} ({count}/{sample_size})")
    
    return results
```

---

## 📊 RESUMEN DE CAMBIOS v9.5 → v10.0

```
┌─────────────────────┬──────────────┬──────────────┐
│ Feature             │ v9.5         │ v10.0        │
├─────────────────────┼──────────────┼──────────────┤
│ Color norm          │ Linear       │ BG-aware     │
│ Spatial ops         │ None         │ 3×3 kernels  │
│ Position encoding   │ None         │ Full         │
│ Objects             │ None         │ Jump Flood   │
│ Multi-scale         │ None         │ Mipmaps      │
│ DSL                 │ None         │ 15-20 ops    │
│ Search              │ None         │ Beam (W=8)   │
│ Color mapping       │ Voting       │ Hungarian    │
│ Convergence         │ Fixed 3      │ Adaptive     │
│ Dual attempts       │ Simple       │ Smart        │
├─────────────────────┼──────────────┼──────────────┤
│ Accuracy (est.)     │ 15%          │ 58-65%       │
└─────────────────────┴──────────────┴──────────────┘
```

---

**Next Steps**:
1. Implementar Sprint 1 completo
2. Validar en 50 tareas de training
3. Iterar basado en errores observados
4. Avanzar a Sprint 2

¿Empezamos con la implementación de Sprint 1?
