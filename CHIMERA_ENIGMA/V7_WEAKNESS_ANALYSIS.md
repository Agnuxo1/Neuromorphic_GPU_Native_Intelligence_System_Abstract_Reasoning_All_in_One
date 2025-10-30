# CHIMERA v7 - Análisis Detallado de Puntos Débiles

## Resumen Ejecutivo

v7 (arquitectura biológica con memoria) es **la arquitectura correcta según el paper** pero tiene **5 puntos críticos** que causan 0% accuracy. Analizamos cada uno y proponemos soluciones.

---

## 1️⃣ PUNTO DÉBIL #1: Codificación de Training (líneas 441-459)

### El Problema
```python
def _encode_training(self, training_examples: List[Dict]) -> np.ndarray:
    # Apila verticalmente los ejemplos de training
    grid = np.zeros((max_h * len(outputs), max_w), dtype=np.uint8)
    for i, output in enumerate(outputs):
        h, w = output.shape
        grid[i*max_h:(i+1)*max_h-max_h+h, :w] = output
```

**Issue**: Pierde información importante:
- Mezcla inputs y outputs sin distinción
- La estructura vertical no preserva relaciones temporales
- Pierde el PATRÓN entre input→output en cada frame

### Solución
Codificar **PARES input/output** separadamente:
```python
def _encode_training_properly(training_examples):
    # Mantener información: "en input A → output B"
    # Mejor: guardar AMBOS y su transformación
    for i, example in enumerate(training_examples):
        inp = example['input']      # Qué vemos
        out = example['output']     # Qué se genera
        # Almacenar: memoria['frame_0'] = (inp, out)
        # Esto permite al CA aprender: inp → out
```

---

## 2️⃣ PUNTO DÉBIL #2: Evolución sin Orientación (líneas 310-342)

### El Problema
```python
def evolve_with_memory(self, frame, memory_key, steps=20):
    for step in range(steps):
        self._render_ca_pass(frame)      # CA evoluciona random
        blended = blend_with_memory()    # Mezcla con memoria
```

**Issue**: El CA no sabe QUÉ evolucionar hacía:
- El CA es **unsupervised** - solo aplica regla de mayoría inercial
- No tiene objetivo de transformación (input→output)
- La memoria solo "mantiene" pero no "guía"

### Solución
Evolución **dirigida** con loss function:
```python
def evolve_with_supervision(frame, training_pairs, memory, steps=20):
    for step in range(steps):
        # 1. Aplicar CA
        evolved = apply_ca(frame)

        # 2. Calcular loss respecto a ejemplos training
        # loss = distancia(evolved_output, training_output)
        loss = compute_training_loss(evolved, training_pairs)

        # 3. Ajustar según loss (gradient-like)
        if loss > threshold:
            # Guiar CA hacia dirección correcta
            evolved = evolved * (1 - loss_influence) + training_output * loss_influence

        # 4. Blend con memoria
        blended = 0.85 * evolved + 0.15 * memory

        # 5. Actualizar frame y memoria
        frame.update(blended)
        memory.update(blended)
```

---

## 3️⃣ PUNTO DÉBIL #3: CA sin Estructura Semántica (líneas 247-297)

### El Problema
El fragment shader solo hace mayoría inercial:
```glsl
// Cuenta vecinos, elige el color que aparece más
int max_count = 0;
int winner = center_color;
for (int i = 0; i < 10; i++) {
    if (counts[i] > max_count) {
        winner = i;
    }
}
```

**Issue**: Es un CA **generic** sin entender ARC:
- No detecta objetos
- No aprende transformaciones (color mapping, scaling, rotation)
- No diferencia "estructura importante" de "ruido"

### Solución
CA con **operadores semánticos** para ARC:

```python
class SemanticCA:
    """CA que entiende transformaciones ARC"""

    def detect_objects(state):
        """Extrae objetos (regiones conectadas)"""
        # Para cada color, identifica clusters conectados
        # Permite evolucionar objetos como unidades

    def apply_transformation(obj, rule):
        """Aplica transformación conocida"""
        # Color mapping: mapear color A → B
        # Scaling: agrandar/empequeñecer objeto
        # Rotation: rotar objeto
        # Translation: mover objeto

    def evolve_semantic(state, memory):
        # No mayoría inercial random
        # Aplica transformaciones que aprendió de training
        objects = detect_objects(state)
        for obj in objects:
            transformation = memory.recall_for(obj)
            obj = apply_transformation(obj, transformation)
        return reconstruct(state, objects)
```

---

## 4️⃣ PUNTO DÉBIL #4: Memoria sin Asociación (líneas 37-150)

### El Problema
```python
def store_pattern(self, name, pattern):
    # Guarda el patrón completo en una textura
    texture = context.texture(pattern)  # Una textura = un patrón

def blend_with_memory(evolved, memory_key):
    # Recupera el patrón y mezcla por píxel
    memory = recall_pattern(memory_key)
    result = 0.85 * evolved + 0.15 * memory
```

**Issue**: Memoria es solo "mezcla píxel a píxel":
- No almacena **transformaciones** (input→output)
- No aprende reglas como "color 1→2"
- No diferencia "recuerdos" por contexto

### Solución
Memoria como **base de transformaciones**:

```python
class SemanticMemory:
    """Memoria que almacena transformaciones"""

    def __init__(self):
        self.color_rules = {}      # color 1 → 2 en training
        self.size_rules = {}       # tamaño patrón: h,w → h',w'
        self.position_rules = {}   # posición objeto
        self.rotation_rules = {}   # rotaciones vistas

    def learn_from_training(training_pairs):
        for inp, out in training_pairs:
            # Aprender: qué colores mapean
            colors = self.detect_color_mapping(inp, out)
            self.color_rules.update(colors)

            # Aprender: cómo cambia tamaño
            size = self.detect_size_change(inp.shape, out.shape)
            self.size_rules.append(size)

            # Aprender: qué se rota/refleja
            rotation = self.detect_rotation(inp, out)
            if rotation:
                self.rotation_rules.append(rotation)

    def apply_learned_rules(test_input):
        """Aplica lo que aprendió"""
        output = test_input.copy()

        # 1. Aplicar mapeos de color conocidos
        for old_color, new_color in self.color_rules.items():
            output[output == old_color] = new_color

        # 2. Aplicar transformación de tamaño
        if self.size_rules:
            target_h, target_w = self.predict_size()
            output = resize(output, (target_h, target_w))

        # 3. Aplicar rotación si se detectó patrón
        if self.rotation_rules:
            output = rotate(output, self.rotation_rules[0])

        return output
```

---

## 5️⃣ PUNTO DÉBIL #5: Sin Validación de Outputs (líneas 393-439)

### El Problema
```python
def solve_task(self, task):
    for test_case in task['test']:
        test_input = np.array(test_case['input'])

        # Evoluciona el test_input ciegamente
        solution = sol_frame.download_state()

        # Retorna directamente sin validar
        solutions.append([solution.tolist(), solution.tolist()])
```

**Issue**: Ninguna validación:
- ¿Es válida la salida? (dimensiones, colores 0-9)
- ¿Es consistente con training? (usa mismos colores)
- ¿Mejora respecto a identidad? (¿es mejor que input sin cambios?)

### Solución
Validación y fallback:

```python
def validate_and_fallback(solution, test_input, training_examples):
    """Valida y propone alternativas si es mala"""

    # 1. Validación básica ARC
    if not is_valid_grid(solution):
        # Solución inválida - usar fallback
        print(f"[WARN] Invalid output shape or colors")
        return apply_identity_rules(test_input, training_examples)

    # 2. Coherencia con training
    sol_colors = set(solution.flatten())
    train_colors = set()
    for ex in training_examples:
        train_colors.update(ex['output'].flatten())

    if not sol_colors.issubset(train_colors):
        # Usa colores no vistos en training
        print(f"[WARN] New colors not in training")
        return apply_identity_rules(test_input, training_examples)

    # 3. Improvement check
    if np.array_equal(solution, test_input):
        # Output es idéntico al input (no cambió nada)
        print(f"[WARN] Solution is identical to input")
        return apply_learner_rules(test_input, training_examples)

    return solution

def apply_identity_rules(test_input, training_examples):
    """Si el CA falla, aplicar lo que definitivamente aprendimos"""
    # Aplicar color mappings detectados en training
    output = test_input.copy()
    color_map = detect_color_mappings(training_examples)
    for old, new in color_map.items():
        output[test_input == old] = new
    return output
```

---

## Resumen de Soluciones

| Punto Débil | Causa | Solución | Impacto |
|------------|-------|---------|--------|
| #1: Training encoding | Pierde pares input/output | Codificar (inp, out) separadamente | Memoria aprende transformaciones |
| #2: Evolución sin dirección | CA no supervisionado | Agregar loss function, guía training | CA evoluciona hacia soluciones correctas |
| #3: CA genérico | No entiende ARC | Operadores semánticos (color, size, rotate) | CA sabe qué transformaciones aplicar |
| #4: Memoria simple | Mezcla píxel a píxel | Almacenar transformaciones explícitas | Memoria actúa como base de conocimiento |
| #5: Sin validación | Retorna outputs sin checks | Validar + fallback a reglas aprendidas | Garantiza soluciones válidas |

---

## Plan de Implementación v7.1

### Fase 1: Memoria Semántica (1-2h)
- Extraer reglas de color mapping de training
- Guardar transformaciones de tamaño
- Detectar rotaciones/reflexiones

### Fase 2: CA Supervisado (2-3h)
- Agregar loss function respecto a training
- Guiar evolución mediante gradientes simulados
- Mantener blending 85/15

### Fase 3: Validación y Fallback (1h)
- Validar outputs
- Si falla CA, aplicar reglas aprendidas

### Fase 4: Testing (1-2h)
- Benchmark v7.1 vs v7 vs v5.3
- Ajustar pesos, ratios, steps

**Expectativa**: v7.1 debería alcanzar **5-10% accuracy** si implementa correctamente la guía supervisionada.

---

## Conclusión

v7 es arquitecturalmente correcta pero implementativamente incompleta:
- ❌ CA evoluciona sin dirección
- ❌ Memoria es pasiva (solo mezcla)
- ❌ No valida outputs
- ❌ Sin aprendizaje de transformaciones

**Con estas 5 soluciones, v7 podría ser la arquitectura ganadora.**
