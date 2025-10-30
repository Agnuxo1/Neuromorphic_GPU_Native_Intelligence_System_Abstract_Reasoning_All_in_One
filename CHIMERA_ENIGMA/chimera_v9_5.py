#!/usr/bin/env python3
"""
CHIMERA v9.5 - Pattern Decoder with Neuromorphic Loop

"No es AGI, es criptografía" - Alan Turing approach to ARC

Key insights:
1. ARC = IQ test temporal sequences (reloj: 12:00, 12:15, 12:30 → 12:45)
2. GPU engañada: renderiza colores que SON los cálculos
3. Loop neuromorphic: Estado + Memoria + Resultado en MISMO fotograma
4. Pattern decoding: Como Enigma, no razonamiento

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import numpy as np
import moderngl
from typing import List, Dict, Tuple, Optional
import time
from collections import Counter


class TemporalPatternDecoder:
    """
    Decodifica patrones temporales como en test IQ.
    """

    @staticmethod
    def detect_size_pattern(examples: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> str:
        """
        Detecta el patrón temporal de tamaños.
        Returns: 'constant', 'identity', 'scale', 'arithmetic', 'geometric'
        """
        if not examples:
            return 'identity'

        in_sizes = [inp for inp, _ in examples]
        out_sizes = [out for _, out in examples]

        # Constant: todos los outputs iguales (como fotograma fijo)
        if len(set(out_sizes)) == 1:
            return 'constant'

        # Identity: input == output (sin transformación)
        if all(inp == out for inp, out in examples):
            return 'identity'

        # Arithmetic progression: diferencia constante
        if len(out_sizes) >= 2:
            diffs_h = [out_sizes[i+1][0] - out_sizes[i][0] for i in range(len(out_sizes)-1)]
            diffs_w = [out_sizes[i+1][1] - out_sizes[i][1] for i in range(len(out_sizes)-1)]
            if len(set(diffs_h)) == 1 and len(set(diffs_w)) == 1:
                return 'arithmetic'

        # Geometric progression: ratio constante
        ratios_h = []
        ratios_w = []
        for i in range(len(in_sizes)):
            if in_sizes[i][0] > 0 and in_sizes[i][1] > 0:
                ratios_h.append(out_sizes[i][0] / in_sizes[i][0])
                ratios_w.append(out_sizes[i][1] / in_sizes[i][1])

        if ratios_h and ratios_w:
            if len(set([round(r, 2) for r in ratios_h])) == 1 and \
               len(set([round(r, 2) for r in ratios_w])) == 1:
                return 'scale'

        return 'identity'

    @staticmethod
    def predict_next_in_sequence(sizes: List[Tuple[int, int]], pattern: str) -> Tuple[int, int]:
        """
        Predice el siguiente tamaño en la secuencia temporal.
        Como predecir 12:45 después de 12:00, 12:15, 12:30
        """
        if not sizes:
            return (3, 3)

        if pattern == 'constant':
            return sizes[-1]

        if pattern == 'arithmetic' and len(sizes) >= 2:
            # Diferencia constante: next = last + diff
            diff_h = sizes[-1][0] - sizes[-2][0]
            diff_w = sizes[-1][1] - sizes[-2][1]
            return (sizes[-1][0] + diff_h, sizes[-1][1] + diff_w)

        if pattern == 'geometric' and len(sizes) >= 2:
            # Ratio constante: next = last * ratio
            ratio_h = sizes[-1][0] / sizes[-2][0] if sizes[-2][0] > 0 else 1
            ratio_w = sizes[-1][1] / sizes[-2][1] if sizes[-2][1] > 0 else 1
            return (int(sizes[-1][0] * ratio_h), int(sizes[-1][1] * ratio_w))

        return sizes[-1]


class NeuromorphicFrame:
    """
    Fotograma neuromorphic: Estado + Memoria + Resultado en UNA textura.

    "todo está en la imagen renderizada, estado, resultado y memoria"
    """

    def __init__(self, ctx: moderngl.Context, size: Tuple[int, int]):
        self.ctx = ctx
        self.h, self.w = size

        # UNA SOLA textura contiene TODO
        # R = estado actual (grid colors)
        # G = memoria (patrones aprendidos)
        # B = resultado emergente
        # A = confianza
        self.unified_texture = ctx.texture(
            size=(self.w, self.h),
            components=4,
            dtype='f4'
        )

        # Inicializar
        data = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.unified_texture.write(data.tobytes())

    def upload_state(self, grid: np.ndarray):
        """Subir estado al fotograma"""
        h, w = grid.shape
        data = np.zeros((self.h, self.w, 4), dtype=np.float32)
        data[:h, :w, 0] = grid.astype(float) / 9.0  # R = estado
        data[:h, :w, 3] = 1.0  # A = confianza inicial
        self.unified_texture.write(data.tobytes())

    def download_result(self) -> np.ndarray:
        """Bajar resultado del fotograma"""
        rgba = np.frombuffer(self.unified_texture.read(), dtype=np.float32)
        rgba = rgba.reshape((self.h, self.w, 4))
        # B channel = resultado emergente
        result = (rgba[:, :, 2] * 9.0).round().astype(np.uint8)
        return result

    def get_texture(self):
        return self.unified_texture

    def release(self):
        self.unified_texture.release()


class LivingBrainV95:
    """
    v9.5: Pattern decoder con loop neuromorphic
    """

    def __init__(self):
        print("="*80)
        print("CHIMERA v9.5 - PATTERN DECODER")
        print("="*80)
        print("'No es AGI, es criptografía' - Turing approach")
        print("Loop neuromorphic: Estado + Memoria + Resultado = 1 fotograma")
        print("="*80)

        # GPU permanente
        self.ctx = moderngl.create_standalone_context()
        print(f"[GPU] {self.ctx.info['GL_RENDERER']}")

        # Memoria global persistente (256x256)
        self.global_memory = self.ctx.texture(
            size=(256, 256), components=4, dtype='f4'
        )
        zeros = np.zeros((256, 256, 4), dtype=np.float32)
        self.global_memory.write(zeros.tobytes())

        print(f"[MEMORY] Global persistent memory: 256x256")

        # Pattern decoder
        self.pattern_decoder = TemporalPatternDecoder()

        # Compile shaders
        self._compile_neuromorphic_shader()

        # Stats
        self.tasks_processed = 0
        self.birth_time = time.time()

        print(f"[BRAIN] v9.5 awakened - Pattern decoder ready")
        print("="*80)

    def _compile_neuromorphic_shader(self):
        """
        Shader neuromorphic: Loop que combina estado + memoria + resultado
        en UN SOLO fotograma.
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

        # Shader neuromorphic: TODO en un fotograma
        fragment_shader = """
        #version 330
        uniform sampler2D u_state;        // Estado actual (R channel)
        uniform sampler2D u_memory;       // Memoria global (RGBA)
        uniform int u_color_map[10];      // Transformación aprendida
        uniform ivec2 grid_size;
        uniform float u_evolution_step;   // Paso evolutivo (0-1)

        in vec2 uv;
        out vec4 out_frame;

        void main() {
            ivec2 coord = ivec2(uv * grid_size);
            coord = clamp(coord, ivec2(0), grid_size - ivec2(1));

            // Leer estado actual (R channel)
            vec4 state_pixel = texelFetch(u_state, coord, 0);
            int input_color = int(state_pixel.r * 9.0 + 0.5);
            input_color = clamp(input_color, 0, 9);

            // Aplicar transformación (pattern decoding)
            int output_color = u_color_map[input_color];
            output_color = clamp(output_color, 0, 9);

            // Leer memoria global
            vec4 memory = texture(u_memory, uv);

            // FOTOGRAMA NEUROMORPHIC:
            // R = estado actual
            // G = memoria (acumulada)
            // B = resultado emergente
            // A = confianza

            float state_val = float(input_color) / 9.0;
            float result_val = float(output_color) / 9.0;
            float memory_val = memory.g * (1.0 - u_evolution_step) + result_val * u_evolution_step;
            float confidence = state_pixel.a;

            out_frame = vec4(
                state_val,      // R: estado
                memory_val,     // G: memoria evolutiva
                result_val,     // B: resultado
                confidence      // A: confianza
            );
        }
        """

        self.neuromorphic_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        print("[SHADER] Neuromorphic loop shader compiled")

    def _decode_pattern(self, train_examples: List[Dict]) -> Tuple[List[int], str, float]:
        """
        Decodifica el patrón (como Enigma, no razonamiento).
        Returns: (color_map, pattern_type, confidence)
        """
        if not train_examples:
            return list(range(10)), 'identity', 0.0

        # Analizar tamaños
        size_pairs = []
        for ex in train_examples:
            in_s = np.array(ex['input']).shape
            out_s = np.array(ex['output']).shape
            size_pairs.append((in_s, out_s))

        # Detectar patrón temporal de tamaños
        pattern_type = self.pattern_decoder.detect_size_pattern(size_pairs)

        # Aprender mapeo de colores
        color_map = list(range(10))
        mapping_counts = {}
        total_consistent = 0
        total_colors = 0

        for ex in train_examples:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)

            if inp.shape != out.shape:
                continue

            for y in range(inp.shape[0]):
                for x in range(inp.shape[1]):
                    old_c = int(inp[y, x])
                    new_c = int(out[y, x])

                    if old_c not in mapping_counts:
                        mapping_counts[old_c] = Counter()
                    mapping_counts[old_c][new_c] += 1

        # Construir mapeo
        for old_c in range(10):
            if old_c in mapping_counts and mapping_counts[old_c]:
                most_common = mapping_counts[old_c].most_common(1)[0]
                new_c, count = most_common
                color_map[old_c] = new_c

                total_colors += 1
                total_for_color = sum(mapping_counts[old_c].values())
                if count / total_for_color > 0.8:
                    total_consistent += 1

        confidence = total_consistent / total_colors if total_colors > 0 else 0.0

        return color_map, pattern_type, confidence

    def _neuromorphic_evolution(self, frame: NeuromorphicFrame, color_map: List[int],
                                steps: int = 3) -> NeuromorphicFrame:
        """
        Evolución neuromorphic: múltiples fotogramas que refinan el resultado.
        """
        current_tex = frame.get_texture()

        for step in range(steps):
            # Crear output texture
            output_tex = self.ctx.texture(
                size=(frame.w, frame.h), components=4, dtype='f4'
            )
            fbo = self.ctx.framebuffer(color_attachments=[output_tex])

            # Aplicar shader neuromorphic
            self.neuromorphic_program['u_state'] = 0
            self.neuromorphic_program['u_memory'] = 1
            self.neuromorphic_program['u_color_map'].write(
                np.array(color_map, dtype='i4').tobytes()
            )
            self.neuromorphic_program['grid_size'] = (frame.w, frame.h)
            self.neuromorphic_program['u_evolution_step'] = (step + 1) / steps

            current_tex.use(location=0)
            self.global_memory.use(location=1)

            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            self._render_quad(self.neuromorphic_program)

            # Ping-pong: output becomes input
            if step < steps - 1:
                current_tex = output_tex
            else:
                # Último paso: actualizar frame
                frame.unified_texture = output_tex

            fbo.release()

        return frame

    def solve_task(self, task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
        """
        Decodifica patrones como Enigma.
        """
        self.tasks_processed += 1

        if verbose:
            age = time.time() - self.birth_time
            print(f"\n[v9.5] Task #{self.tasks_processed} | Age: {age:.1f}s")

        start = time.time()

        # Decodificar patrón
        color_map, pattern_type, confidence = self._decode_pattern(task['train'])

        if verbose:
            if color_map != list(range(10)):
                mappings = {i: color_map[i] for i in range(10) if color_map[i] != i}
                print(f"[DECODE] Pattern: {pattern_type}")
                print(f"[DECODE] Mappings: {mappings}")
                print(f"[DECODE] Confidence: {confidence:.2%}")

        # Predecir tamaños de salida
        size_pairs = [(np.array(ex['input']).shape, np.array(ex['output']).shape)
                      for ex in task['train']]

        predictions = []
        for test_case in task['test']:
            test_input = np.array(test_case['input'], dtype=np.uint8)

            # Predecir tamaño (temporal pattern)
            out_sizes = [out for _, out in size_pairs]
            predicted_size = self.pattern_decoder.predict_next_in_sequence(
                out_sizes, pattern_type
            )

            think_start = time.time()

            # Crear fotograma neuromorphic
            frame = NeuromorphicFrame(self.ctx, predicted_size)
            frame.upload_state(test_input)

            # Evolución neuromorphic (3 pasos)
            frame = self._neuromorphic_evolution(frame, color_map, steps=3)

            # Extraer resultado
            result = frame.download_result()

            think_time = time.time() - think_start

            if verbose:
                print(f"[THINK] Evolution: 3 steps | Time: {think_time*1000:.2f}ms")

            # Dual attempt
            attempt1 = result
            if confidence < 0.5:
                # Baja confianza: intento con identity
                frame2 = NeuromorphicFrame(self.ctx, predicted_size)
                frame2.upload_state(test_input)
                frame2 = self._neuromorphic_evolution(frame2, list(range(10)), steps=1)
                attempt2 = frame2.download_result()
                frame2.release()
            else:
                attempt2 = result.copy()

            predictions.append([
                attempt1.tolist(),
                attempt2.tolist()
            ])

            frame.release()

        total_time = time.time() - start
        if verbose:
            print(f"[TOTAL] Time: {total_time*1000:.1f}ms")

        return predictions

    def _render_quad(self, program):
        """Render fullscreen quad"""
        vertices = np.array([
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.simple_vertex_array(program, vbo, 'in_vert')
        vao.render(moderngl.TRIANGLE_STRIP)
        vao.release()
        vbo.release()

    def get_stats(self):
        age = time.time() - self.birth_time
        return {
            'version': '9.5',
            'tasks_processed': self.tasks_processed,
            'age_seconds': age,
            'alive': True
        }

    def __del__(self):
        if hasattr(self, 'global_memory'):
            self.global_memory.release()
        if hasattr(self, 'ctx'):
            self.ctx.release()


_global_brain_v95 = None

def get_brain_v95():
    global _global_brain_v95
    if _global_brain_v95 is None:
        _global_brain_v95 = LivingBrainV95()
    return _global_brain_v95

def solve_arc_task(task: Dict, verbose: bool = True):
    brain = get_brain_v95()
    return brain.solve_task(task, verbose=verbose)


if __name__ == "__main__":
    print("="*80)
    print("CHIMERA v9.5 - LOCAL TEST")
    print("="*80)

    # Test: Temporal sequence (como reloj)
    task = {
        'train': [
            {'input': [[1, 2], [3, 4]], 'output': [[2, 3], [4, 5]]},
            {'input': [[5, 6], [7, 8]], 'output': [[6, 7], [8, 9]]},
        ],
        'test': [
            {'input': [[1, 2], [3, 4]]}
        ]
    }

    brain = get_brain_v95()
    result = brain.solve_task(task)

    print(f"\nResult: {result[0][0]}")
    print(f"Expected: [[2, 3], [4, 5]]")
    print(f"Match: {result[0][0] == [[2, 3], [4, 5]]}")

    stats = brain.get_stats()
    print(f"\nv9.5 Stats: {stats}")
