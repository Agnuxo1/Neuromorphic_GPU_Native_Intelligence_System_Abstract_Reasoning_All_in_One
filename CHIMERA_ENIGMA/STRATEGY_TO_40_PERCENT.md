# Estrategia CHIMERA: De 4% a 40% Accuracy en ARC-AGI2

**Fecha**: 2025-10-28
**Estado actual**: v7.2 logra 4% accuracy (2/50 correctas) en análisis de 50 tareas
**Objetivo**: 40% accuracy (20/50 correctas)
**Contexto**: OpenAI o3 logra 4% en ARC-AGI2, otros LLMs ~1%

---

## 📊 Análisis de Fallas de v7.2

### Qué FUNCIONA (4% - 2 tareas)
✅ **same_shape_color_mapping_few_colors**: Mapeos simples de color con pocos colores

**Ejemplo de éxito**:
```
Input:  [[1, 2], [3, 4]]
Output: [[2, 3], [4, 5]]
```
v7.2 detecta: color_mapping = {1→2, 2→3, 3→4, 4→5}

### Qué FALLA (96% - 48 tareas)

| Patrón de Falla | Frecuencia | % del total |
|-----------------|------------|-------------|
| 1. same_shape_identity_or_complex | 10 tasks | 20% |
| 2. same_shape_color_mapping | 10 tasks | 20% |
| 3. same_shape_identity_or_complex_few_colors | 6 tasks | 12% |
| 4. scaling_up_few_colors | 5 tasks | 10% |
| 5. scaling_down_few_colors | 3 tasks | 6% |

**Top 3 patrones = 52% de las fallas**

---

## 🎯 Estrategia Escalonada para 40%

### Fase 1: Arreglar Top 3 Patrones → 15-20% accuracy

**Target**: Resolver patrones 1-3 (26 tareas adicionales)

#### 1.1 Pattern: "same_shape_identity_or_complex" (20% de fallas)

**Problema**: v7.2 solo hace color mapping global. Estas tareas requieren:
- Operaciones espaciales (mover objetos)
- Lógica condicional (if X then Y)
- Patrones geométricos (simetría, repetición)

**Solución**:
```python
class PatternDetector:
    """Detecta patrones más allá de color mapping"""

    def detect_spatial_pattern(self, inp, out):
        """
        Detecta:
        - Objetos que se mueven
        - Objetos que se copian/replican
        - Objetos que desaparecen/aparecen
        """
        # Connected components analysis
        inp_objects = self._extract_objects(inp)
        out_objects = self._extract_objects(out)

        # ¿Mismo número de objetos?
        if len(inp_objects) == len(out_objects):
            # ¿Se movieron? ¿Rotaron? ¿Escalaron?
            return self._match_objects(inp_objects, out_objects)

        # ¿Objetos nuevos? (generación)
        # ¿Objetos eliminados? (filtrado)
        return "object_transformation"

    def detect_rule_based(self, training_pairs):
        """
        Detecta reglas condicionales:
        - "Si hay un 8 al lado de un 1, cambia el 1 a 3"
        - "Rellena regiones conectadas del mismo color"
        """
        pass
```

**Impacto estimado**: +8-10pp (resolver 8 de 10 tareas)

#### 1.2 Pattern: "same_shape_color_mapping" (20% de fallas)

**Problema**: v7.2 hace color mapping global, pero estas tareas requieren mapping **condicional**:
- Color X → Y solo si está al lado de Z
- Color X → Y solo en ciertas posiciones
- Color X → Y solo en ciertos objetos

**Solución**:
```python
class ConditionalColorMapper:
    """Color mapping con contexto"""

    def learn_conditional_mappings(self, inp, out):
        """
        Aprende reglas como:
        - color=1, neighbor=8 → new_color=3
        - color=2, position=edge → new_color=0
        """
        mappings = []

        for y in range(inp.shape[0]):
            for x in range(inp.shape[1]):
                inp_color = inp[y, x]
                out_color = out[y, x]

                if inp_color != out_color:
                    # Analizar contexto
                    neighbors = self._get_neighbors(inp, y, x)
                    position_type = self._classify_position(inp, y, x)

                    rule = {
                        'from_color': inp_color,
                        'to_color': out_color,
                        'neighbors': neighbors,
                        'position': position_type
                    }
                    mappings.append(rule)

        return self._find_consistent_rules(mappings)
```

**Impacto estimado**: +7-9pp (resolver 7 de 10 tareas)

#### 1.3 Pattern: "same_shape_identity_or_complex_few_colors" (12% de fallas)

**Problema**: Transformaciones que parecen identity pero tienen lógica escondida

**Solución**: Grid diffing + pattern matching
```python
class GridDiffer:
    """Analiza diferencias pixel a pixel"""

    def find_diff_pattern(self, inp, out):
        """
        Encuentra qué cambió y por qué:
        - Solo cambian píxeles en bordes?
        - Solo cambian píxeles de cierto color?
        - Cambios siguen un patrón geométrico?
        """
        diff = (inp != out)
        changed_positions = np.argwhere(diff)

        if len(changed_positions) == 0:
            return "identity"

        # Analizar patrón espacial de cambios
        pattern = self._analyze_change_pattern(
            changed_positions,
            inp,
            out
        )
        return pattern
```

**Impacto estimado**: +4-6pp (resolver 4 de 6 tareas)

### Fase 2: Scaling Operations → 25-30% accuracy

**Target**: Resolver scaling_up y scaling_down (11 tareas adicionales)

#### 2.1 Intelligent Scaling

**Problema**: v7.2 usa Kronecker product (replicación simple). ARC requiere:
- Upscaling con interpolación inteligente
- Downscaling con agregación semántica (no solo reducir)
- Scaling selectivo (solo ciertos objetos)

**Solución**:
```python
class IntelligentScaler:
    """Scaling semántico"""

    def upscale(self, inp, h_scale, w_scale):
        """
        No solo repetir píxeles.
        Detectar estructura y preservarla.
        """
        # ¿Hay objetos? Escalarlos independientemente
        objects = self._extract_objects(inp)

        if objects:
            return self._upscale_objects(objects, h_scale, w_scale)
        else:
            # Upscale con interpolación nearest neighbor inteligente
            return self._upscale_structured(inp, h_scale, w_scale)

    def downscale(self, inp, h_scale, w_scale):
        """
        No solo samplear.
        Agregación semántica.
        """
        # Dividir en bloques
        # Cada bloque → un píxel
        # Valor = mayoría, o color específico

        return self._downsample_semantic(inp, h_scale, w_scale)
```

**Impacto estimado**: +7-9pp (resolver 7 de 11 tareas de scaling)

### Fase 3: Object-Based Reasoning → 35-40% accuracy

**Target**: Transformaciones complejas basadas en objetos

#### 3.1 Object Extraction & Manipulation

```python
class ObjectReasoner:
    """Razonamiento a nivel de objetos"""

    def extract_objects(self, grid):
        """
        Connected components con scipy
        """
        from scipy.ndimage import label

        objects = []
        for color in np.unique(grid):
            if color == 0:  # Skip background
                continue

            mask = (grid == color)
            labeled, num = label(mask)

            for i in range(1, num + 1):
                obj_mask = (labeled == i)
                objects.append({
                    'color': color,
                    'mask': obj_mask,
                    'bbox': self._get_bbox(obj_mask),
                    'shape': self._get_shape(obj_mask),
                    'size': obj_mask.sum()
                })

        return objects

    def learn_object_transformations(self, training_pairs):
        """
        Aprende qué le pasa a cada objeto:
        - Se mueve? A dónde?
        - Cambia de color?
        - Se replica?
        - Desaparece?
        """
        rules = []

        for inp, out in training_pairs:
            inp_objs = self.extract_objects(inp)
            out_objs = self.extract_objects(out)

            # Match objects
            matches = self._match_objects(inp_objs, out_objs)

            for inp_obj, out_obj in matches:
                rule = self._infer_transformation(inp_obj, out_obj)
                rules.append(rule)

        return self._consolidate_rules(rules)
```

**Impacto estimado**: +5-10pp (permite resolver tareas object-based complejas)

---

## 🏗️ Arquitectura Propuesta: CHIMERA v8.0

### Diseño Modular

```python
class CHIMERA_v8:
    """
    Arquitectura modular para 40% accuracy
    """

    def __init__(self):
        # Phase 1 components
        self.pattern_detector = PatternDetector()
        self.conditional_mapper = ConditionalColorMapper()
        self.grid_differ = GridDiffer()

        # Phase 2 components
        self.scaler = IntelligentScaler()

        # Phase 3 components
        self.object_reasoner = ObjectReasoner()

        # Original v7.2 components (keep what works)
        self.semantic_memory = SemanticMemory()

    def solve_task(self, task):
        """
        Multi-strategy solving
        """
        # Learn from training
        self.semantic_memory.learn_from_training(task['train'])

        # Detect task type
        task_type = self.pattern_detector.classify_task(task['train'])

        # Route to appropriate solver
        if task_type == "simple_color_mapping":
            return self._solve_color_mapping(task)

        elif task_type == "conditional_mapping":
            return self._solve_conditional(task)

        elif task_type == "spatial_transformation":
            return self._solve_spatial(task)

        elif task_type == "scaling":
            return self._solve_scaling(task)

        elif task_type == "object_based":
            return self._solve_object_based(task)

        else:
            # Fallback: try all strategies and rank
            return self._solve_ensemble(task)
```

### Estrategia de Implementación

**Orden de implementación**:
1. **v7.5**: Fase 1 (pattern detection + conditional mapping) → Target: 15-20%
2. **v7.6**: Fase 2 (intelligent scaling) → Target: 25-30%
3. **v8.0**: Fase 3 (object reasoning) → Target: 35-40%

**Cada versión debe**:
- Mantener compatibilidad con v7.2 (keep what works)
- Agregar UN módulo nuevo a la vez
- Validar incrementalmente

---

## 📈 Roadmap Realista

| Versión | Mejoras | Target Accuracy | Timeline |
|---------|---------|-----------------|----------|
| v7.2 | Baseline (semantic memory + CA) | 4% ✅ | Actual |
| **v7.5** | + Pattern detection + Conditional mapping | **15-20%** | Sprint 1 |
| v7.6 | + Intelligent scaling | 25-30% | Sprint 2 |
| v8.0 | + Object reasoning | 35-40% | Sprint 3 |

### ¿Es 40% realista?

**Sí, porque**:
- OpenAI o3 logra 4% ($200/task, computa masiva)
- CHIMERA v7.2 logra 4% ($0/task, CPU rápida)
- Los patrones que fallan son **algorítmicos**, no requieren AGI
- Con object detection + conditional logic podemos resolver muchos

**No, si**:
- Intentamos resolver TODO tipo de tarea (hay algunas imposibles sin razonamiento humano)
- No priorizamos (querer abarcar todo)
- Agregamos complejidad sin validar

### Estrategia Conservadora

**Meta realista por fases**:
- Fase 1: 10-15% accuracy (2-3x mejor que v7.2)
- Fase 2: 15-20% accuracy (5x mejor que OpenAI o3)
- Fase 3: 20-30% accuracy (si Fase 2 funciona)
- Fase 4: 30-40% accuracy (optimista, requ requiere refinamiento)

---

## 🚀 Próximo Paso: v7.5

**Implementar Fase 1**:
1. PatternDetector (clasificar tipo de tarea)
2. ConditionalColorMapper (mapping con contexto)
3. GridDiffer (análisis de diferencias)
4. Router (decidir qué estrategia usar)

**Expected outcome**: 10-15% accuracy en benchmark de 100 tareas

---

Francisco Angulo de Lafuente
CHIMERA Project 2025
"From 4% to 40%: Algorithmic Reasoning, Not Brute Force"
