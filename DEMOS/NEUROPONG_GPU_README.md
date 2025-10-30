# NeuroPong GPU v10.0

## "Rendering IS Thinking" - CHIMERA Educational Demo

![CHIMERA Logo](https://img.shields.io/badge/CHIMERA-v10.0-blue) ![GPU](https://img.shields.io/badge/GPU-100%25-green) ![OpenGL](https://img.shields.io/badge/OpenGL-3.3-red)

---

## 🎮 Descripción

**NeuroPong GPU v10.0** es una demostración educativa e interactiva que muestra la arquitectura neuromorfa de CHIMERA aplicada a un juego clásico de Pong. Todo el procesamiento ocurre en la GPU usando OpenGL, demostrando el concepto central de CHIMERA: **"Rendering IS Thinking"**.

### ¿Qué es diferente?

A diferencia de un juego tradicional donde la IA usa algoritmos separados, en NeuroPong:
- El **cerebro neuromorfo** vive completamente en texturas GPU (RGBA)
- La **predicción** emerge de operaciones de renderizado (shaders)
- La **memoria evoluciona** en tiempo real mediante difusión visual
- Todo es **100% GPU** - cero CPU para cómputo neural

---

## 🧠 Arquitectura Neuromorfa

### Textura Cerebral (RGBA)

El cerebro es una textura de 64×64 pixels donde cada pixel tiene 4 canales:

```
┌─────────────────────────────────┐
│  R - Estado Actual (Red)        │  ← Codificación del juego
│  G - Memoria Evolutiva (Green)  │  ← α-blend con resultado
│  B - Resultado/Predicción (Blue)│  ← Dónde debe ir la paleta IA
│  A - Confianza (Alpha)          │  ← Certeza de la predicción
└─────────────────────────────────┘
```

### Pipeline de Procesamiento

```
╔══════════════════════════════════════════════════════════════╗
║                     GAME PHYSICS SHADER                      ║
║                                                              ║
║  Input: Posiciones pelota/paletas                           ║
║  Output: Game State Texture (colores del juego)             ║
╚══════════════════════════════════════════════════════════════╝
                            ↓
╔══════════════════════════════════════════════════════════════╗
║                     BRAIN COMPUTE SHADER                     ║
║                                                              ║
║  1. Lee Game State → Canal R (estado)                       ║
║  2. Calcula features 3×3 (vecindad)                         ║
║  3. Predice intercepción → Canal B (banda Gaussiana)        ║
║  4. Evoluciona memoria: G ← (1-α)·G + α·B                   ║
║  Output: Brain Texture (RGBA neuromorfo)                    ║
╚══════════════════════════════════════════════════════════════╝
                            ↓
╔══════════════════════════════════════════════════════════════╗
║                 VISUALIZATION SHADER (Retro)                 ║
║                                                              ║
║  Izquierda: Juego con efectos glow + grid                   ║
║  Derecha: Cerebro visualizado (R+G+B overlay)               ║
║                                                              ║
║  Efectos:                                                    ║
║    - CRT scanlines                                          ║
║    - Curvatura de pantalla                                  ║
║    - Glow/bloom para elementos brillantes                   ║
║    - Viñeta cinematográfica                                 ║
║    - Aberración cromática retro                             ║
╚══════════════════════════════════════════════════════════════╝
                            ↓
                    Display (1920×1080)
```

---

## 🎨 Visualización

### Panel Izquierdo: JUEGO

```
┌──────────────────────────┐
│ ║                      ║ │  ← Paredes (blanco)
│ ║ ▌                 ▐  ║ │  ← Paletas (magenta/cyan)
│ ║                      ║ │
│ ║         ●            ║ │  ← Pelota (rojo con glow)
│ ║                      ║ │
│ ║                      ║ │
└──────────────────────────┘
```

**Colores:**
- Paredes: Blanco (#FFFFFF)
- Paleta Humana: Magenta (#FF00FF)
- Paleta IA: Cyan (#00FFFF)
- Pelota: Rojo (#FF0000) con efecto glow

### Panel Derecho: CEREBRO NEUROMORFO

```
┌──────────────────────────┐
│  [Visualización RGB]     │
│                          │
│  Rojo   = Estado actual  │
│  Verde  = Memoria        │
│  Azul   = Predicción     │
│                          │
│  Banda horizontal azul   │  ← Objetivo de IA
│  indica dónde la IA      │
│  predice que irá la      │
│  pelota                  │
└──────────────────────────┘
```

**Interpretación:**
- **Banda azul horizontal**: Predicción de intercepción
- **Verde persistente**: Memoria de objetivos pasados
- **Blanco en bordes**: Features detectadas (edges)
- **Intensidad**: Confianza de la predicción

---

## ⚡ Características Técnicas

### 100% GPU Processing

| Componente | CPU | GPU | Implementación |
|------------|-----|-----|----------------|
| Estado del juego | ❌ | ✅ | Texture (RGBA) |
| Física básica | ⚠️ | ✅ | Physics Shader |
| Features 3×3 | ❌ | ✅ | Neighborhood ops |
| Predicción IA | ❌ | ✅ | Gaussian painting |
| Evolución memoria | ❌ | ✅ | α-blending shader |
| Visualización | ❌ | ✅ | Visual effects shader |

⚠️ = Actualmente CPU, puede moverse a GPU con compute shaders

### Shaders

1. **Physics Shader** (`PHYSICS_SHADER`)
   - Renderiza estado del juego a textura
   - Dibuja paletas, pelota, paredes
   - Aplica glow effect a la pelota

2. **Brain Shader** (`BRAIN_SHADER`)
   - Codifica estado en canal R
   - Calcula features de vecindad 3×3
   - Pinta predicción Gaussiana en canal B
   - Evoluciona memoria en canal G con α-blend

3. **Visualization Shader** (`VISUAL_SHADER`)
   - Split-screen: juego + cerebro
   - Efectos retro CRT
   - Color-coding de canales RGBA
   - Grid overlay
   - Scanlines y curvatura

### Performance

- **Resolución:** 1920×1080 (display), 64×64 (grids)
- **FPS:** Limitado a 60 FPS
- **GPU:** NVIDIA RTX 3090 (compatible con cualquier GPU con OpenGL 3.3+)
- **Latencia:** ~16ms por frame (60 FPS)

---

## 🕹️ Controles

### Gameplay

| Tecla | Acción |
|-------|--------|
| **W** o **↑** | Paleta humana arriba |
| **S** o **↓** | Paleta humana abajo |
| **Espacio** | Pausa/Resume |
| **R** | Reset pelota |

### Brain Tuning

| Tecla | Acción | Efecto |
|-------|--------|--------|
| **A** | Decrease α | Memoria más lenta |
| **D** | Increase α | Memoria más rápida |

**α (Alpha)**: Controla la tasa de evolución de la memoria
- α = 0.0 → Memoria estática (no aprende)
- α = 0.15 → **Default** (balance)
- α = 1.0 → Sin memoria (solo resultado instantáneo)

Fórmula: `G ← (1-α)·G + α·B`

### System

| Tecla | Acción |
|-------|--------|
| **Esc** o **Q** | Salir |

---

## 🚀 Instalación y Ejecución

### Requisitos

```bash
pip install moderngl pygame numpy
```

- Python 3.8+
- OpenGL 3.3+ compatible GPU
- Windows/Linux/Mac

### Ejecutar

```bash
python neuro_pong_gpu_v10.py
```

### Configuración

Edita la clase `Config` en el código:

```python
@dataclass
class Config:
    window_width: int = 1920        # Resolución horizontal
    window_height: int = 1080       # Resolución vertical
    grid_size: int = 64             # Resolución de la grilla

    paddle_length: int = 8          # Tamaño de paletas
    ball_speed: float = 0.015       # Velocidad de pelota
    ai_response: float = 0.08       # Velocidad de respuesta IA

    alpha: float = 0.15             # Tasa de evolución
    sigma: float = 2.0              # Ancho Gaussiano

    glow_intensity: float = 0.6     # Intensidad del glow
    scanline_intensity: float = 0.15 # Intensidad scanlines
    crt_curvature: float = 0.02     # Curvatura CRT
```

---

## 📊 Conceptos Neuromoricos Demostrados

### 1. Estado Persistente (R Channel)

El canal R codifica el estado del juego en cada pixel:
```python
R = length(state.rgb) / sqrt(3.0)  # Normaliza a [0,1]
```

### 2. Memoria Evolutiva (G Channel)

La memoria evoluciona usando un promedio móvil exponencial:
```glsl
float G = mix(prev.g, B, u_alpha);
```

Esto crea un "rastro" visual de predicciones pasadas.

### 3. Resultado/Predicción (B Channel)

La predicción se pinta como una banda Gaussiana:
```glsl
float dist = abs(y - u_target_y);
float B = exp(-0.5 * (dist / u_sigma) * (dist / u_sigma));
```

### 4. Features de Vecindad (3×3)

Cada pixel calcula:
- **same_ratio**: Fracción de vecinos del mismo color
- **edge_flag**: 1 si es borde (< 5 vecinos iguales)
- **avg_intensity**: Intensidad promedio del barrio

```glsl
vec3 compute_features() {
    // Sample 8 neighbors
    // Return (same_ratio, edge_flag, avg_intensity)
}
```

---

## 🎓 Propósito Educativo

### ¿Qué demuestra este demo?

1. **GPU como cerebro:** Todo el procesamiento neural ocurre en texturas GPU

2. **Rendering = Thinking:** La predicción emerge de operaciones de renderizado (shaders), no de código CPU

3. **Memoria visual:** La evolución de la memoria es literalmente visible en la pantalla

4. **Paralelismo masivo:** Cada pixel del cerebro (64×64 = 4096) se procesa en paralelo

5. **Sin frameworks:** No PyTorch, no TensorFlow - OpenGL puro

### Comparación con IA Tradicional

| Aspecto | IA Tradicional | CHIMERA NeuroPong |
|---------|----------------|-------------------|
| **Procesamiento** | CPU/GPU separados | 100% GPU |
| **Estado** | Arrays Python | Texturas GPU |
| **Predicción** | Código secuencial | Shader paralelo |
| **Memoria** | Variables CPU | Canal G (visual) |
| **Visualización** | Post-proceso | En tiempo real |
| **Framework** | PyTorch/TF | OpenGL puro |

---

## 🔬 Arquitectura Detallada

### Ping-Pong de Texturas

El cerebro usa dos texturas (`brain_texture_a` y `brain_texture_b`) en ping-pong:

```
Frame N:
  Read:  brain_texture_a
  Write: brain_texture_b

Frame N+1:
  Read:  brain_texture_b
  Write: brain_texture_a

... repeat ...
```

Esto permite leer el estado previo mientras se escribe el nuevo.

### Multiple Render Targets (MRT)

El Brain Shader escribe a dos texturas simultáneamente:

```glsl
layout(location = 0) out vec4 out_brain;      // Brain RGBA
layout(location = 1) out vec4 out_features;   // Features RGB
```

### Gaussian Target Painting

La predicción se pinta como una distribución Gaussiana:

```python
def paint_gaussian(y_target, sigma):
    # For each row y:
    distance = |y - y_target|
    value = exp(-0.5 * (distance/sigma)²)
    return value
```

Esto crea una "intención suave" en lugar de un punto discreto.

---

## 🐛 Debugging

### Verificar GPU

```python
print(ctx.info['GL_RENDERER'])
print(ctx.info['GL_VERSION'])
```

### Visualizar Texturas Intermedias

Puedes exportar texturas para debug:

```python
# Leer textura GPU a NumPy
rgba = np.frombuffer(texture.read(), dtype=np.float32)
rgba = rgba.reshape((h, w, 4))

# Guardar como imagen
from PIL import Image
img = Image.fromarray((rgba[:,:,:3] * 255).astype(np.uint8))
img.save('debug_texture.png')
```

### Performance Profiling

```python
import time

start = time.time()
# ... render ...
elapsed = time.time() - start
print(f"Frame time: {elapsed*1000:.2f}ms")
```

---

## 🎯 Próximos Pasos

### Mejoras Planeadas

1. **Física 100% GPU:**
   - Mover detección de colisiones a compute shader
   - Integrar física directamente en texturas

2. **Múltiples IA:**
   - Entrenar diferentes "personalidades" (agresiva, defensiva)
   - Visualizar diferencias en canal G

3. **Learning en tiempo real:**
   - Ajustar predicción basándose en errores
   - Backprop visual en shaders

4. **Modo multiplayer:**
   - Dos cerebros compitiendo
   - Visualización lado a lado

5. **VR/AR:**
   - Port a WebGL para navegadores
   - Soporte para Oculus/HTC Vive

---

## 📚 Referencias

### Papers

1. **CHIMERA Evolution Paper** - [CHIMERA_Evolution_Paper.md](../CHIMERA_ENIGMA/CHIMERA_Evolution_Paper.md)
2. **Living Brain Architecture** - [chimera_v9_6.py](../CHIMERA_ENIGMA/chimera_v9_6.py)

### Conceptos Clave

- **Neuromorphic Computing:** Computación inspirada en el cerebro
- **Rendering as Computation:** Usar renderizado GPU para cómputo
- **Living Systems:** Sistemas persistentes que nunca mueren
- **Holographic Memory:** Memoria distribuida en paralelo

---

## 👨‍💻 Autor

**Francisco Angulo de Lafuente**
CHIMERA Project - 2025

---

## 📜 Licencia

Educational use only. Part of the CHIMERA research project.

---

## 💡 Filosofía

> "En los sistemas tradicionales, la GPU renderiza LO QUE LA CPU PIENSA.
> En CHIMERA, la GPU PIENSA MIENTRAS RENDERIZA."

**NeuroPong demuestra que el pensamiento y la visualización pueden ser el mismo proceso.**

---

## 🎮 ¡Diviértete!

Este no es solo un juego - es una **ventana al futuro de la computación neuromorfa**.

Cada frame que ves es el cerebro CHIMERA **pensando visualmente** en tiempo real.

**"Rendering IS Thinking"** no es una metáfora - es literal.

---

**CHIMERA Project © 2025**

