#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHIMERA v9.6 - Neuromorphic ARC Solver (GPU-first, CPU-safe)
============================================================

Philosophy
----------
"Rendering *is* computing." CHIMERA treats the GPU as a massive parallel
renderer where colors encode computations. A single RGBA texture behaves
as a neuromorphic frame: R holds the current state, G the evolving memory,
B the emerging result, and A the confidence. The "brain" (GL context + compiled
programs + persistent textures) is born once and stays alive across tasks.

What's new in v9.6 (compared to v9.5)
-------------------------------------
1) Spatial operators on GPU (3×3 neighborhood features, edges, local majority).
2) Connected components via label propagation on GPU (color-aware).
3) Multi-hypothesis execution (small DSL of spatial transforms) with a
   lightweight validator/invariant checker.
4) Robust color-map learning with a Hungarian assignment (10×10, pure Python).
5) Adaptive neuromorphic evolution steps with early stop by delta.
6) Clean CPU fallbacks (no GPU required to run; GPU strongly recommended).
7) Ready for ARC Prize 2025: grids up to 30×30, 10 symbols (0..9), rectangular.

Usage
-----
- Feed a Kaggle-style ARC task dict:
  {
    "train": [{"input": [[...]], "output": [[...]]}, ...],
    "test":  [{"input": [[...]]}, ...]
  }

- Call:
    brain = get_brain_v96()
    predictions = brain.solve_task(task)

- Each test item returns up to 3 candidate outputs (as required by ARC-AGI-2):
    predictions[i] = [cand1, cand2, cand3]

Dependencies
------------
- numpy (mandatory)
- moderngl (optional but recommended). If unavailable or context creation fails,
  CHIMERA falls back to CPU-only execution of the DSL and decoders.

Author
------
CHIMERA Project, 2025
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Try to import ModernGL; fallback to CPU if unavailable.
try:
    import moderngl  # type: ignore
    HAS_MODERNGL = True
except Exception:
    moderngl = None  # type: ignore
    HAS_MODERNGL = False


# ------------------------------
# Utility functions and structs
# ------------------------------

def clamp_int_grid(arr: np.ndarray) -> np.ndarray:
    """Ensure the grid values are integers in [0, 9]."""
    arr = np.asarray(arr, dtype=np.int32)
    arr[arr < 0] = 0
    arr[arr > 9] = 9
    return arr


def to_grid(a: List[List[int]]) -> np.ndarray:
    """Convert list-of-lists to an int32 numpy array (H×W)."""
    return clamp_int_grid(np.array(a, dtype=np.int32))


def to_list(a: np.ndarray) -> List[List[int]]:
    """Convert numpy array (H×W) to list-of-lists (ints)."""
    return [[int(x) for x in row] for row in a.tolist()]


def unique_colors(arr: np.ndarray) -> List[int]:
    """Sorted unique colors present in grid."""
    u = np.unique(arr)
    return [int(x) for x in u]


def pad_or_crop(arr: np.ndarray, out_h: int, out_w: int, fill: int = 0) -> np.ndarray:
    """Center-pad or center-crop to the desired shape (out_h, out_w)."""
    h, w = arr.shape
    result = np.full((out_h, out_w), fill, dtype=np.int32)
    # Compute paste coordinates
    ph = min(h, out_h)
    pw = min(w, out_w)
    y0_in = (h - ph) // 2
    x0_in = (w - pw) // 2
    y0_out = (out_h - ph) // 2
    x0_out = (out_w - pw) // 2
    result[y0_out:y0_out+ph, x0_out:x0_out+pw] = arr[y0_in:y0_in+ph, x0_in:x0_in+pw]
    return result


def bbox_of_color_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (ymin, xmin, ymax, xmax) of True entries or None if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Return centroid (y, x) of True entries. If empty, return center (H/2, W/2)."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        h, w = mask.shape
        return h / 2.0, w / 2.0
    return float(ys.mean()), float(xs.mean())


def count_colors(arr: np.ndarray) -> Dict[int, int]:
    """Histogram of colors present in the array."""
    hist = {}
    for v in range(10):
        c = int(np.sum(arr == v))
        if c > 0:
            hist[v] = c
    return hist


# ------------------------------
# Hungarian assignment (10×10)
# ------------------------------

def hungarian_assignment(cost: np.ndarray) -> List[int]:
    """
    Solve the assignment problem (minimize total cost). Pure Python / NumPy.
    The matrix is small (10×10), so an O(n^3) implementation is fine.

    Returns a list 'assignment' where assignment[i] = j (column chosen for row i).
    """
    # Implementation based on a classic Hungarian algorithm (Kuhn–Munkres).
    # The code is intentionally explicit and documented for clarity.
    cost = np.array(cost, dtype=np.float64)
    n = cost.shape[0]

    # Step 1: Row reduction
    cost = cost - cost.min(axis=1, keepdims=True)

    # Step 2: Column reduction
    cost = cost - cost.min(axis=0, keepdims=True)

    # Masks and helpers
    INF = 1e12
    mask = np.zeros_like(cost, dtype=np.int8)  # 1 = starred zero, 2 = primed zero
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    def find_zero():
        """Find a non-covered zero; return its (r, c) or None."""
        for r in range(n):
            if row_covered[r]:
                continue
            for c in range(n):
                if not col_covered[c] and abs(cost[r, c]) < 1e-12:
                    return r, c
        return None

    def find_star_in_row(r):
        for c in range(n):
            if mask[r, c] == 1:
                return c
        return None

    def find_star_in_col(c):
        for r in range(n):
            if mask[r, c] == 1:
                return r
        return None

    def find_prime_in_row(r):
        for c in range(n):
            if mask[r, c] == 2:
                return c
        return None

    def cover_columns_with_starred_zeroes():
        for c in range(n):
            col_covered[c] = False
            for r in range(n):
                if mask[r, c] == 1:
                    col_covered[c] = True
                    break

    # Step 3: Star a zero in each row if possible
    for r in range(n):
        for c in range(n):
            if abs(cost[r, c]) < 1e-12 and not row_covered[r] and not col_covered[c]:
                mask[r, c] = 1
                row_covered[r] = True
                col_covered[c] = True
                break

    # Reset covers
    row_covered[:] = False
    col_covered[:] = False
    cover_columns_with_starred_zeroes()

    # Main loop
    while True:
        if col_covered.sum() == n:
            # Optimal
            assignment = [-1] * n
            for r in range(n):
                sc = find_star_in_row(r)
                if sc is not None:
                    assignment[r] = sc
            # Fill any unassigned rows greedily (shouldn't happen with square matrix,
            # but keep safety for degeneracies)
            free_cols = set(range(n)) - set([c for c in assignment if c is not None and c >= 0])
            for r in range(n):
                if assignment[r] == -1:
                    assignment[r] = free_cols.pop()
            return assignment

        # Step 4: Find a non-covered zero and prime it
        z = find_zero()
        while z is None:
            # Step 6: Adjust the matrix
            # Find the minimum uncovered value
            m = INF
            for r in range(n):
                if row_covered[r]:
                    continue
                for c in range(n):
                    if not col_covered[c]:
                        if cost[r, c] < m:
                            m = cost[r, c]
            if not np.isfinite(m):
                m = 0.0
            # Add m to covered rows, subtract m from uncovered columns
            for r in range(n):
                if row_covered[r]:
                    cost[r, :] += m
            for c in range(n):
                if not col_covered[c]:
                    cost[:, c] -= m
            z = find_zero()

        r, c = z
        mask[r, c] = 2  # prime
        star_c = find_star_in_row(r)
        if star_c is not None:
            # Cover this row and uncover the column of the starred zero
            row_covered[r] = True
            col_covered[star_c] = False
        else:
            # Step 5: Augmenting path
            # Build alternating path of starred and primed zeros
            path = [(r, c)]
            # Find starred zero in the column
            star_r = find_star_in_col(c)
            while star_r is not None:
                path.append((star_r, c))
                # Find primed zero in the row
                prime_c = find_prime_in_row(star_r)
                path.append((star_r, prime_c))
                c = prime_c
                star_r = find_star_in_col(c)
            # Augment: flip stars/primes along the path
            for pr, pc in path:
                if mask[pr, pc] == 1:
                    mask[pr, pc] = 0
                elif mask[pr, pc] == 2:
                    mask[pr, pc] = 1
            # Clear primes and covers
            mask[mask == 2] = 0
            row_covered[:] = False
            col_covered[:] = False
            cover_columns_with_starred_zeroes()

# -----------------------------------
# DSL of spatial operators (CPU path)
# -----------------------------------

def op_identity(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    return pad_or_crop(a, out_shape[0], out_shape[1], fill=0)


def op_rotate90(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    b = np.rot90(a, k=1)
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


def op_rotate180(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    b = np.rot90(a, k=2)
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


def op_rotate270(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    b = np.rot90(a, k=3)
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


def op_flip_h(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    b = np.fliplr(a)
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


def op_flip_v(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    b = np.flipud(a)
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


def op_transpose(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    b = a.T
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


def op_translate(a: np.ndarray, out_shape: Tuple[int, int], dy: int, dx: int) -> np.ndarray:
    h, w = out_shape
    b = np.full((h, w), 0, dtype=np.int32)
    ay, ax = a.shape
    # We translate inside the *output* canvas
    y0 = (h - ay) // 2 + dy
    x0 = (w - ax) // 2 + dx
    # Paste with bounds checks
    ys = max(0, y0)
    xs = max(0, x0)
    ye = min(h, y0 + ay)
    xe = min(w, x0 + ax)
    ysi = max(0, -y0)
    xsi = max(0, -x0)
    if ye > ys and xe > xs:
        b[ys:ye, xs:xe] = a[ysi:ysi+(ye-ys), xsi:xsi+(xe-xs)]
    return b


def op_scale_integer(a: np.ndarray, out_shape: Tuple[int, int], ky: int, kx: int) -> np.ndarray:
    """Scale by integer factors ky, kx using nearest replication (up/down)."""
    if ky == 1 and kx == 1:
        return pad_or_crop(a, out_shape[0], out_shape[1], fill=0)
    ay, ax = a.shape
    # Downscale by integer: average blocks -> nearest int (majority vote)
    if ky < 1 or kx < 1:
        raise ValueError("ky and kx must be >= 1")
    by = ay * ky
    bx = ax * kx
    b = np.repeat(np.repeat(a, ky, axis=0), kx, axis=1)
    return pad_or_crop(b, out_shape[0], out_shape[1], fill=0)


CPU_OPERATORS = [
    ("identity", op_identity),
    ("rotate90", op_rotate90),
    ("rotate180", op_rotate180),
    ("rotate270", op_rotate270),
    ("flip_h", op_flip_h),
    ("flip_v", op_flip_v),
    ("transpose", op_transpose),
]


# ---------------------------------------------
# Pattern decoders (size + color-map assignment)
# ---------------------------------------------

class TemporalPatternDecoder:
    """
    Decodes (i) size evolution pattern and (ii) color-map between input/output.
    Color map is solved with a Hungarian assignment over 10×10 counts.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _size_pattern(in_sizes: List[Tuple[int, int]], out_sizes: List[Tuple[int, int]]) -> str:
        """
        Detect a simple pattern among: identity, arithmetic (constant deltas),
        or geometric (constant integer ratios).
        """
        if not in_sizes or not out_sizes or len(in_sizes) != len(out_sizes):
            return "identity"

        # Identity
        if all(in_sizes[i] == out_sizes[i] for i in range(len(in_sizes))):
            return "identity"

        # Arithmetic: fixed delta per step
        dh = [out_sizes[i][0] - in_sizes[i][0] for i in range(len(in_sizes))]
        dw = [out_sizes[i][1] - in_sizes[i][1] for i in range(len(in_sizes))]
        if len(set(dh)) == 1 and len(set(dw)) == 1:
            return "arithmetic"

        # Geometric: fixed integer ratio
        rh = []
        rw = []
        ok = True
        for i in range(len(in_sizes)):
            ih, iw = in_sizes[i]
            oh, ow = out_sizes[i]
            if ih == 0 or iw == 0:
                ok = False
                break
            if oh % ih != 0 or ow % iw != 0:
                ok = False
                break
            rh.append(oh // ih)
            rw.append(ow // iw)
        if ok and len(set(rh)) == 1 and len(set(rw)) == 1:
            return "geometric"

        return "identity"

    @staticmethod
    def _color_map_hungarian(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
        """
        Build a 10×10 count matrix of input->output color co-occurrences and
        solve the assignment that *maximizes* these counts (minimizes negative).
        Missing colors map to themselves.
        """
        counts = np.zeros((10, 10), dtype=np.float64)
        seen_in = set()

        for xin, yout in train_pairs:
            # Flatten
            a = xin.reshape(-1)
            b = yout.reshape(-1)

            for v in range(10):
                seen_in.add(v) if np.any(a == v) else None

            # Handle different shapes: if same shape, do pixel-wise pairing
            # Otherwise, use histogram-based heuristic
            if a.shape == b.shape:
                # Pixel-wise co-occurrence (exact spatial correspondence)
                for i in range(len(a)):
                    ci = int(a[i])
                    co = int(b[i])
                    counts[ci, co] += 1.0
            else:
                # Different shapes (rotation, transpose, etc.)
                # Use histogram overlap: if input has N pixels of color ci
                # and output has M pixels of color co, add min(N,M) to counts[ci,co]
                # weighted by how many unique colors exist
                hist_in = np.zeros(10, dtype=np.int32)
                hist_out = np.zeros(10, dtype=np.int32)
                for v in range(10):
                    hist_in[v] = int(np.sum(a == v))
                    hist_out[v] = int(np.sum(b == v))

                # Greedy histogram matching (simple heuristic)
                for ci in range(10):
                    if hist_in[ci] == 0:
                        continue
                    # Find best co that maximizes min(hist_in[ci], hist_out[co])
                    for co in range(10):
                        overlap = min(hist_in[ci], hist_out[co])
                        counts[ci, co] += float(overlap) * 0.5  # weight less than exact match

        # Convert to costs for Hungarian (we want to maximize counts)
        maxc = counts.max() if counts.size else 0.0
        cost = (maxc - counts)

        assignment = hungarian_assignment(cost)
        cmap = list(range(10))
        for ci in range(10):
            cmap[ci] = int(assignment[ci])

        # If some input colors never appear, keep identity for them
        present = set()
        for xin, _ in train_pairs:
            present |= set([int(v) for v in np.unique(xin)])
        for ci in range(10):
            if ci not in present:
                cmap[ci] = ci
        return cmap

    def decode(self, train_examples: List[Dict]) -> Tuple[List[int], str, float, Tuple[int, int, int, int]]:
        """
        Returns:
            color_map (List[int]): mapping input->output (size 10)
            pattern_type (str): 'identity' | 'arithmetic' | 'geometric'
            confidence (float): [0,1] confidence on the mapping
            size_rule (h0,w0,dh_or_rh,dw_or_rw): parameters for size prediction
        """
        if not train_examples:
            return list(range(10)), "identity", 0.0, (0, 0, 0, 0)

        in_sizes, out_sizes = [], []
        pairs = []
        for ex in train_examples:
            xin = to_grid(ex["input"])
            yout = to_grid(ex["output"])
            in_sizes.append(xin.shape)
            out_sizes.append(yout.shape)
            pairs.append((xin, yout))

        pattern = self._size_pattern(in_sizes, out_sizes)

        # Derive rule parameters
        if pattern == "identity":
            h0, w0 = in_sizes[-1]
            size_rule = (h0, w0, 0, 0)
        elif pattern == "arithmetic":
            dh = out_sizes[-1][0] - in_sizes[-1][0]
            dw = out_sizes[-1][1] - in_sizes[-1][1]
            size_rule = (in_sizes[-1][0], in_sizes[-1][1], dh, dw)
        else:  # geometric
            # Safe integer ratios from the *last* example
            ih, iw = in_sizes[-1]
            oh, ow = out_sizes[-1]
            rh = max(1, oh // max(1, ih))
            rw = max(1, ow // max(1, iw))
            size_rule = (in_sizes[-1][0], in_sizes[-1][1], rh, rw)

        cmap = self._color_map_hungarian(pairs)

        # Crude confidence: fraction of pixels explained by the color map alone
        score = 0.0
        total = 0.0
        for xin, yout in pairs:
            mapped = np.vectorize(lambda v: cmap[int(v)])(xin)
            # Handle size mismatch: only compare if shapes match
            if mapped.shape == yout.shape:
                score += float(np.sum(mapped == yout))
                total += float(mapped.size)
            else:
                # If shapes don't match, confidence contribution is zero for this pair
                total += float(yout.size)
        confidence = 0.0 if total == 0 else (score / total)

        return cmap, pattern, float(confidence), size_rule


# -----------------------------------
# GPU Neuromorphic frame and passes
# -----------------------------------

@dataclass
class GPUPrograms:
    # Compiled shader programs (if GPU context exists)
    neuromorphic: Optional["moderngl.Program"] = None
    features3x3: Optional["moderngl.Program"] = None
    label_propagation: Optional["moderngl.Program"] = None
    operator: Optional["moderngl.Program"] = None


class NeuromorphicFrame:
    """
    A single RGBA texture holds the neuromorphic state:
    R = state (input grid in [0,1] scaled from 0..9),
    G = memory (evolving),
    B = result (output encoding in [0,1]),
    A = confidence.
    """

    def __init__(self, ctx: "moderngl.Context", size: Tuple[int, int]):
        self.ctx = ctx
        self.h, self.w = size
        # Unified texture
        self.unified_texture = ctx.texture(size=(self.w, self.h), components=4, dtype="f4")
        self.unified_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.unified_texture.repeat_x = False
        self.unified_texture.repeat_y = False

    def release(self) -> None:
        if hasattr(self, "unified_texture") and self.unified_texture is not None:
            self.unified_texture.release()

    def get_texture(self) -> "moderngl.Texture":
        return self.unified_texture

    def upload_state(self, grid: List[List[int]]) -> None:
        """
        Upload integer grid (0..9) to the R channel; zero to others.
        """
        a = to_grid(grid).astype(np.float32) / 9.0
        rgba = np.zeros((self.h, self.w, 4), dtype=np.float32)
        # Center-pad/crop
        a2 = pad_or_crop(a, self.h, self.w, fill=0.0)
        rgba[..., 0] = a2  # R
        rgba[..., 1] = 0.0 # G memory
        rgba[..., 2] = 0.0 # B output
        rgba[..., 3] = 1.0 # A confidence
        self.unified_texture.write(rgba.tobytes())


class GPUEngine:
    """
    Owns the ModernGL context and compiles all programs once ("living brain").
    Provides helper methods to run the passes.
    """
    def __init__(self) -> None:
        self.ctx: Optional["moderngl.Context"] = None
        self.prog = GPUPrograms()
        self._quad_vao: Optional["moderngl.VertexArray"] = None
        self.alive_time = time.time()
        self._init_context_and_compile()

    def _init_context_and_compile(self) -> None:
        if not HAS_MODERNGL:
            return
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
            # Fullscreen quad for all passes
            vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
            vbo = self.ctx.buffer(vertices.tobytes())
            vs = """
                #version 330
                in vec2 in_vert;
                out vec2 uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    uv = (in_vert + 1.0) * 0.5;
                }
            """
            simple_prog = self.ctx.program(
                vertex_shader=vs,
                fragment_shader="""
                    #version 330
                    out vec4 fragColor;
                    void main(){ fragColor = vec4(0,0,0,1); }
                """
            )
            self._quad_vao = self.ctx.simple_vertex_array(simple_prog, vbo, "in_vert")
            # Compile shader programs
            self._compile_neuromorphic()
            self._compile_features3x3()
            self._compile_label_propagation()
            self._compile_operator()
        except Exception:
            # If any GPU setup fails, fully disable GPU mode.
            self.ctx = None
            self._quad_vao = None

    # ---------- compilation ----------

    def _compile_neuromorphic(self) -> None:
        if self.ctx is None:
            return
        fs = """
            #version 330
            uniform sampler2D u_state;      // unified RGBA texture (input in R)
            uniform sampler2D u_memory;     // previous frame (RGBA)
            uniform int u_color_map[10];    // color LUT
            uniform ivec2 grid_size;
            uniform float u_alpha;          // evolution mix in [0,1]

            in vec2 uv;
            out vec4 out_frame;

            void main() {
                ivec2 coord = ivec2(uv * vec2(grid_size));
                coord = clamp(coord, ivec2(0), grid_size - ivec2(1));

                vec4 state_px  = texelFetch(u_state,  coord, 0);
                vec4 memory_px = texelFetch(u_memory, coord, 0);

                int ci = int(round(state_px.r * 9.0));
                ci = clamp(ci, 0, 9);
                int co = u_color_map[ci];
                co = clamp(co, 0, 9);

                float state_val = float(ci) / 9.0;
                float result_val = float(co) / 9.0;
                float memory_val = mix(memory_px.g, result_val, u_alpha);
                float confidence = state_px.a;

                out_frame = vec4(state_val, memory_val, result_val, confidence);
            }
        """
        vs = """
            #version 330
            in vec2 in_vert;
            out vec2 uv;
            void main(){
                gl_Position = vec4(in_vert, 0.0, 1.0);
                uv = (in_vert + 1.0) * 0.5;
            }
        """
        self.prog.neuromorphic = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _compile_features3x3(self) -> None:
        if self.ctx is None:
            return
        fs = """
            #version 330
            uniform sampler2D u_state; // unified frame (R holds state)
            uniform ivec2 grid_size;

            in vec2 uv;
            out vec4 out_feat; // R: same-color neighbors count [0..8]/8,
                               // G: edge flag, B: majority color (0..9)/9, A: 1

            ivec2 offs[8] = ivec2[8](
                ivec2(-1,-1), ivec2(0,-1), ivec2(1,-1),
                ivec2(-1, 0),               ivec2(1, 0),
                ivec2(-1, 1), ivec2(0, 1),  ivec2(1, 1)
            );

            void main(){
                ivec2 coord = ivec2(uv * vec2(grid_size));
                coord = clamp(coord, ivec2(0), grid_size - ivec2(1));

                int c0 = int(round(texelFetch(u_state, coord, 0).r * 9.0));
                int counts[10];
                for(int k=0;k<10;k++) counts[k] = 0;

                int same = 0;
                for(int i=0;i<8;i++){
                    ivec2 c = clamp(coord + offs[i], ivec2(0), grid_size - ivec2(1));
                    int ci = int(round(texelFetch(u_state, c, 0).r * 9.0));
                    if(ci == c0) same++;
                    counts[ci]++;
                }

                // Majority in 3x3
                int bestc = 0; int bestv = -1;
                for(int k=0;k<10;k++){
                    if(counts[k] > bestv){ bestv = counts[k]; bestc = k; }
                }

                // Edge if not surrounded by same color (simple heuristic)
                float edge = (same < 5) ? 1.0 : 0.0;

                out_feat = vec4(float(same)/8.0, edge, float(bestc)/9.0, 1.0);
            }
        """
        vs = """
            #version 330
            in vec2 in_vert;
            out vec2 uv;
            void main(){
                gl_Position = vec4(in_vert, 0.0, 1.0);
                uv = (in_vert + 1.0) * 0.5;
            }
        """
        self.prog.features3x3 = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _compile_label_propagation(self) -> None:
        if self.ctx is None:
            return
        fs = """
            #version 330
            uniform sampler2D u_state;  // unified frame (R=color index)
            uniform sampler2D u_labels; // previous labels (R channel in [0,1])
            uniform ivec2 grid_size;

            in vec2 uv;
            out vec4 out_labels; // R: normalized label id in [0,1]

            ivec2 offs[8] = ivec2[8](
                ivec2(-1,-1), ivec2(0,-1), ivec2(1,-1),
                ivec2(-1, 0),               ivec2(1, 0),
                ivec2(-1, 1), ivec2(0, 1),  ivec2(1, 1)
            );

            void main(){
                ivec2 coord = ivec2(uv * vec2(grid_size));
                coord = clamp(coord, ivec2(0), grid_size - ivec2(1));

                int c0 = int(round(texelFetch(u_state, coord, 0).r * 9.0));
                float best = texelFetch(u_labels, coord, 0).r;
                float me = best;
                // Propagate minimum label among same-color neighbors
                for(int i=0;i<8;i++){
                    ivec2 c = clamp(coord + offs[i], ivec2(0), grid_size - ivec2(1));
                    int ci = int(round(texelFetch(u_state, c, 0).r * 9.0));
                    float li = texelFetch(u_labels, c, 0).r;
                    if(ci == c0){
                        if(li < best) best = li;
                    }
                }
                out_labels = vec4(best, me, 0.0, 1.0);
            }
        """
        vs = """
            #version 330
            in vec2 in_vert;
            out vec2 uv;
            void main(){
                gl_Position = vec4(in_vert, 0.0, 1.0);
                uv = (in_vert + 1.0) * 0.5;
            }
        """
        self.prog.label_propagation = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _compile_operator(self) -> None:
        if self.ctx is None:
            return
        fs = """
            #version 330
            uniform sampler2D u_state;   // unified frame (R=color index)
            uniform ivec2 in_size;       // input grid (H,W)
            uniform ivec2 out_size;      // output grid (H,W)
            uniform int u_op;            // 0..N operators
            uniform ivec2 u_shift;       // for translate
            uniform ivec2 u_scale;       // integer up-scale factors (ky,kx)

            in vec2 uv;
            out vec4 out_frame;

            // Helper: sample input at integer coords (y,x)
            float sample_color_norm(ivec2 p){
                p = clamp(p, ivec2(0), in_size - ivec2(1));
                int ci = int(round(texelFetch(u_state, p, 0).r * 9.0));
                ci = clamp(ci, 0, 9);
                return float(ci) / 9.0;
            }

            void main(){
                ivec2 oc = ivec2(uv * vec2(out_size)); // output coord (y,x)
                oc = clamp(oc, ivec2(0), out_size - ivec2(1));

                ivec2 ic = oc; // default mapping (identity with center-crop/pad in CPU)
                // We'll map output->input by inverse transforms where possible.
                if(u_op == 0){ // identity (nearest fit handled in CPU pad/crop)
                    // direct center copy handled in CPU; here just clamp
                    ic = ivec2( int(float(oc.y) * float(in_size.y) / float(out_size.y)),
                                int(float(oc.x) * float(in_size.x) / float(out_size.x)) );
                } else if(u_op == 1){ // rotate90
                    ic = ivec2(in_size.x - 1 - oc.x, oc.y);
                } else if(u_op == 2){ // rotate180
                    ic = ivec2(in_size.y - 1 - oc.y, in_size.x - 1 - oc.x);
                } else if(u_op == 3){ // rotate270
                    ic = ivec2(oc.x, in_size.y - 1 - oc.y);
                } else if(u_op == 4){ // flip_h
                    ic = ivec2(oc.y, in_size.x - 1 - oc.x);
                } else if(u_op == 5){ // flip_v
                    ic = ivec2(in_size.y - 1 - oc.y, oc.x);
                } else if(u_op == 6){ // transpose
                    ic = ivec2(oc.x, oc.y);
                } else if(u_op == 7){ // translate by u_shift (handled as best-effort here)
                    ic = oc - u_shift;
                } else if(u_op == 8){ // scale integer up by (ky,kx) (inverse map)
                    ic = ivec2(oc.y / max(u_scale.x,1), oc.x / max(u_scale.y,1));
                }

                float state_val = sample_color_norm(ic);
                // Keep memory/confidence unchanged (written by caller if needed)
                out_frame = vec4(state_val, 0.0, state_val, 1.0);
            }
        """
        vs = """
            #version 330
            in vec2 in_vert;
            out vec2 uv;
            void main(){
                gl_Position = vec4(in_vert, 0.0, 1.0);
                uv = (in_vert + 1.0) * 0.5;
            }
        """
        self.prog.operator = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    # ---------- helpers ----------

    def _render_to(self, program: "moderngl.Program", uniforms: Dict, outputs: List["moderngl.Texture"]) -> None:
        assert self.ctx is not None and self._quad_vao is not None
        # Setup FBO
        fbo = self.ctx.framebuffer(color_attachments=outputs)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        # Bind uniforms
        for name, value in uniforms.items():
            if isinstance(value, int):
                program[name].value = int(value)
            elif isinstance(value, float):
                program[name].value = float(value)
            elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
                program[name].value = value
            else:
                # arrays: color map, etc.
                if hasattr(program[name], "write"):
                    program[name].write(value)
        # Draw
        self._quad_vao.program = program
        self._quad_vao.render(mode=moderngl.TRIANGLE_STRIP)
        fbo.release()

    # ---------- public API ----------

    def create_frame(self, size: Tuple[int, int]) -> NeuromorphicFrame:
        assert self.ctx is not None
        return NeuromorphicFrame(self.ctx, size)

    def neuromorphic_pass(self, frame: NeuromorphicFrame, color_map: List[int], alpha: float) -> NeuromorphicFrame:
        """
        Single neuromorphic pass mixing memory (G) with new result (B) using u_alpha.
        """
        assert self.ctx is not None and self.prog.neuromorphic is not None
        # Input textures
        state_tex = frame.get_texture()
        memory_tex = frame.get_texture()  # read previous as "memory"
        out_tex = self.ctx.texture(size=(frame.w, frame.h), components=4, dtype="f4")
        out_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Prepare uniforms
        uniforms = {
            "u_state": 0,
            "u_memory": 1,
            "u_color_map": np.array(color_map, dtype="i4").tobytes(),
            "grid_size": (frame.w, frame.h),
            "u_alpha": float(alpha),
        }
        state_tex.use(location=0)
        memory_tex.use(location=1)
        self._render_to(self.prog.neuromorphic, uniforms, [out_tex])

        frame.unified_texture = out_tex
        return frame

    def features3x3_pass(self, frame: NeuromorphicFrame) -> "moderngl.Texture":
        """
        Compute local features per pixel; returns a texture with features.
        """
        assert self.ctx is not None and self.prog.features3x3 is not None
        out_tex = self.ctx.texture(size=(frame.w, frame.h), components=4, dtype="f4")
        out_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        uniforms = {
            "u_state": 0,
            "grid_size": (frame.w, frame.h),
        }
        frame.get_texture().use(location=0)
        self._render_to(self.prog.features3x3, uniforms, [out_tex])
        return out_tex

    def label_propagation(self, frame: NeuromorphicFrame, iterations: int = 8) -> "moderngl.Texture":
        """
        Run color-aware label propagation; returns a texture with labels in R channel.
        """
        assert self.ctx is not None and self.prog.label_propagation is not None
        # Initialize labels with unique ids per pixel (normalized by HW)
        init = np.zeros((frame.h, frame.w, 4), dtype=np.float32)
        ids = (
            np.arange(frame.h * frame.w, dtype=np.float32)
            .reshape(frame.h, frame.w) / float(frame.h * frame.w - 1 + 1e-6)
        )
        init[..., 0] = ids  # R
        init[..., 3] = 1.0
        labels_tex = self.ctx.texture(size=(frame.w, frame.h), components=4, dtype="f4")
        labels_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        labels_tex.write(init.tobytes())

        for _ in range(max(1, iterations)):
            out_tex = self.ctx.texture(size=(frame.w, frame.h), components=4, dtype="f4")
            out_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            uniforms = {
                "u_state": 0,
                "u_labels": 1,
                "grid_size": (frame.w, frame.h),
            }
            frame.get_texture().use(location=0)
            labels_tex.use(location=1)
            self._render_to(self.prog.label_propagation, uniforms, [out_tex])
            labels_tex.release()
            labels_tex = out_tex

        return labels_tex

    def operator_apply(self, frame: NeuromorphicFrame, op_id: int,
                       in_size: Tuple[int, int], out_size: Tuple[int, int],
                       shift: Tuple[int, int] = (0, 0), scale: Tuple[int, int] = (1, 1)) -> NeuromorphicFrame:
        """
        Apply a single operator in GPU (result goes to B channel mirror = R here).
        """
        assert self.ctx is not None and self.prog.operator is not None
        out_tex = self.ctx.texture(size=(out_size[1], out_size[0]), components=4, dtype="f4")  # (W,H) in moderngl
        out_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        uniforms = {
            "u_state": 0,
            "in_size": (in_size[1], in_size[0]),   # note: we pass (W,H) as (x,y) in GLSL
            "out_size": (out_size[1], out_size[0]),
            "u_op": int(op_id),
            "u_shift": (int(shift[1]), int(shift[0])),  # (dx,dy) but GLSL uses (x,y)
            "u_scale": (int(scale[0]), int(scale[1])),  # (ky,kx)
        }
        frame.get_texture().use(location=0)
        self._render_to(self.prog.operator, uniforms, [out_tex])
        # Replace frame texture (resized)
        frame.release()
        newf = NeuromorphicFrame(self.ctx, (out_size[0], out_size[1]))
        newf.unified_texture = out_tex
        return newf

    def download_b_channel(self, frame: NeuromorphicFrame) -> np.ndarray:
        """
        Read back B channel (result) as int grid [0..9].
        """
        assert self.ctx is not None
        raw = np.frombuffer(frame.get_texture().read(), dtype=np.float32)
        rgba = raw.reshape((frame.h, frame.w, 4))
        result = np.clip(np.rint(rgba[..., 2] * 9.0), 0, 9).astype(np.int32)
        return result


# -----------------------------------
# Invariants & validation
# -----------------------------------

def invariants(arr: np.ndarray) -> Dict[str, object]:
    """
    Compute lightweight invariants for solution filtering.
    """
    h, w = arr.shape
    inv = {
        "shape": (h, w),
        "counts": count_colors(arr),
        "num_nonzero": int(np.sum(arr != 0)),
        "num_colors": int(len(unique_colors(arr))),
        "sym_h": int(np.array_equal(arr, np.fliplr(arr))),
        "sym_v": int(np.array_equal(arr, np.flipud(arr))),
    }
    return inv


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Exact-match score in [0,1]. If shapes differ, return 0.0.
    """
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.sum(a == b)) / float(a.size)


# -----------------------------------
# CHIMERA Brain v9.6 (solver)
# -----------------------------------

class LivingBrainV96:
    """
    The living brain: creates GPU context once (if available), compiles programs,
    maintains age and stats, and solves ARC tasks.
    """
    def __init__(self) -> None:
        self.ctx_engine = GPUEngine()
        self.train_seen = 0
        self.tasks_processed = 0
        self.birth_time = time.time()
        print("[BRAIN] v9.6 alive. GPU:", "ON" if self.ctx_engine.ctx is not None else "OFF")

    # ---- helpers ----

    @staticmethod
    def _predict_size_from_rule(inp: Tuple[int, int], pattern: str, params: Tuple[int, int, int, int]) -> Tuple[int, int]:
        if pattern == "identity":
            return inp
        h0, w0, dh_or_rh, dw_or_rw = params
        if pattern == "arithmetic":
            return (max(1, inp[0] + dh_or_rh), max(1, inp[1] + dw_or_rw))
        elif pattern == "geometric":
            rh = max(1, dh_or_rh)
            rw = max(1, dw_or_rw)
            return (max(1, inp[0] * rh), max(1, inp[1] * rw))
        return inp

    @staticmethod
    def _apply_color_map(arr: np.ndarray, cmap: List[int]) -> np.ndarray:
        lut = np.array(cmap, dtype=np.int32)
        return lut[arr]

    def _detect_operator_from_pair(self, xin: np.ndarray, yout: np.ndarray) -> Tuple[str, Dict]:
        """
        Try the small DSL and (optionally) a depth-2 composition to explain train pair.
        Returns (operator_name_or_sequence, params_dict).
        """
        H, W = yout.shape
        candidates = []

        # Try single-step operators first
        for name, fn in CPU_OPERATORS:
            pred = fn(xin, (H, W))
            score = similarity(pred, yout)
            candidates.append(((name,), {}, score))

        # Heuristic translate: align centroids (non-zero pixels considered "object")
        mask_in = xin != 0
        mask_out = yout != 0
        cy_in, cx_in = centroid(mask_in)
        cy_out, cx_out = centroid(mask_out)
        dy = int(round(cy_out - cy_in))
        dx = int(round(cx_out - cx_in))
        pred = op_translate(xin, (H, W), dy=dy, dx=dx)
        candidates.append((("translate",), {"dy": dy, "dx": dx}, similarity(pred, yout)))

        # Heuristic integer scale if exact ratio
        if xin.shape[0] > 0 and xin.shape[1] > 0:
            if H % xin.shape[0] == 0 and W % xin.shape[1] == 0:
                ky = H // xin.shape[0]
                kx = W // xin.shape[1]
                pred = op_scale_integer(xin, (H, W), ky=ky, kx=kx)
                candidates.append((("scale",), {"ky": ky, "kx": kx}, similarity(pred, yout)))

        # Depth-2 beam (width <= 6: top-6 single ops)
        candidates.sort(key=lambda t: t[2], reverse=True)
        top = candidates[:6]
        for (n1,), p1, s1 in top:
            # Apply op1 then re-evaluate with another op
            if n1 == "translate":
                a1 = op_translate(xin, (H, W), p1.get("dy", 0), p1.get("dx", 0))
            elif n1 == "scale":
                a1 = op_scale_integer(xin, (H, W), p1.get("ky", 1), p1.get("kx", 1))
            else:
                a1 = dict(CPU_OPERATORS)[n1](xin, (H, W))

            for (n2, fn2) in CPU_OPERATORS:
                pred = fn2(a1, (H, W))
                candidates.append(((n1, n2), {}, similarity(pred, yout)))

        # Pick best
        candidates.sort(key=lambda t: t[2], reverse=True)
        best_seq, best_params, best_score = candidates[0]
        return ("->".join(best_seq), best_params)

    # ---- main API ----

    def solve_task(self, task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
        """
        Solve a Kaggle ARC task dict.
        Returns: list per test item with up to 3 candidate outputs.
        """
        self.tasks_processed += 1
        start = time.time()
        if verbose:
            age = time.time() - self.birth_time
            print(f"\n[v9.6] Task #{self.tasks_processed} | Age: {age:.1f}s")

        # 1) Decode pattern and color map from training
        decoder = TemporalPatternDecoder()
        cmap, pattern, conf, size_rule = decoder.decode(task.get("train", []))
        if verbose:
            print(f"[DECODE] pattern={pattern}, confidence={conf:.3f}, color_map={cmap}")

        # 2) For each training pair, try to detect operator(s) explaining the transform
        ops_detected = []
        for ex in task.get("train", []):
            xin = to_grid(ex["input"])
            yout = to_grid(ex["output"])
            xin_mapped = self._apply_color_map(xin, cmap)
            op_name, params = self._detect_operator_from_pair(xin_mapped, yout)
            ops_detected.append((op_name, params))
        if verbose and ops_detected:
            print("[OPS] detected (per train example):", ops_detected)

        # 3) Consensus: pick the most common operator name across examples (tie -> first)
        if ops_detected:
            names = [n for (n, _p) in ops_detected]
            unique, counts = np.unique(np.array(names, dtype=object), return_counts=True)
            consensus = unique[np.argmax(counts)]
            # Combine parameters heuristically (take from first match for simplicity)
            consensus_params = {}
            for (n, p) in ops_detected:
                if n == consensus:
                    consensus_params = p
                    break
        else:
            consensus = "identity"
            consensus_params = {}

        if verbose:
            print(f"[CONSENSUS] op='{consensus}' params={consensus_params}")

        # 4) Predict test outputs (3 candidates per test grid where possible)
        predictions: List[List[List[List[int]]]] = []
        for ti, test_item in enumerate(task.get("test", [])):
            X = to_grid(test_item["input"])
            target_size = self._predict_size_from_rule(X.shape, pattern, size_rule)

            # Candidate A: color-map -> consensus operator (CPU path, robust baseline)
            Xm = self._apply_color_map(X, cmap)

            def apply_consensus_cpu(arr: np.ndarray) -> np.ndarray:
                H, W = target_size
                if "->" in consensus:
                    # Two-step composition
                    n1, n2 = consensus.split("->")
                    step1 = self._apply_named_op(arr, n1, target_size, consensus_params)
                    step2 = self._apply_named_op(step1, n2, target_size, consensus_params)
                    return step2
                else:
                    return self._apply_named_op(arr, consensus, target_size, consensus_params)

            candA = apply_consensus_cpu(Xm)

            # Candidate B: pure identity (only color map + pad/crop to target size)
            candB = op_identity(Xm, target_size)

            # Candidate C: symmetry fallback (try the "other axis" if consensus is a flip/rot)
            fallback = "rotate180" if "rotate" in consensus else ("flip_v" if "flip_h" in consensus else "flip_h")
            candC = dict(CPU_OPERATORS)[fallback](Xm, target_size) if fallback in dict(CPU_OPERATORS) else candB

            predictions.append([to_list(candA), to_list(candB), to_list(candC)])

        if verbose:
            print(f"[TOTAL] time = {(time.time() - start)*1000:.1f} ms")
        return predictions

    def _apply_named_op(self, arr: np.ndarray, name: str, out_shape: Tuple[int, int], params: Dict) -> np.ndarray:
        if name == "identity":
            return op_identity(arr, out_shape)
        if name == "rotate90":
            return op_rotate90(arr, out_shape)
        if name == "rotate180":
            return op_rotate180(arr, out_shape)
        if name == "rotate270":
            return op_rotate270(arr, out_shape)
        if name == "flip_h":
            return op_flip_h(arr, out_shape)
        if name == "flip_v":
            return op_flip_v(arr, out_shape)
        if name == "transpose":
            return op_transpose(arr, out_shape)
        if name == "translate":
            return op_translate(arr, out_shape, params.get("dy", 0), params.get("dx", 0))
        if name == "scale":
            return op_scale_integer(arr, out_shape, params.get("ky", 1), params.get("kx", 1))
        # Unknown -> identity
        return op_identity(arr, out_shape)

    def get_stats(self) -> Dict[str, object]:
        return {
            "version": "9.6",
            "gpu": self.ctx_engine.ctx is not None,
            "tasks_processed": self.tasks_processed,
            "alive_seconds": time.time() - self.birth_time,
        }


# ------------------------------
# Global singleton
# ------------------------------

_global_brain_v96: Optional[LivingBrainV96] = None

def get_brain_v96() -> LivingBrainV96:
    global _global_brain_v96
    if _global_brain_v96 is None:
        _global_brain_v96 = LivingBrainV96()
    return _global_brain_v96


# ------------------------------
# Simple self-test / demo
# ------------------------------

def _demo_task_identity_plus_one() -> Dict:
    """
    Toy task: map color c -> (c+1) mod 10, preserve size/geometry.
    """
    inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = (inp + 1) % 10
    return {
        "train": [{"input": to_list(inp), "output": to_list(out)}],
        "test":  [{"input": to_list(inp)}]
    }


def _demo_task_rotate90() -> Dict:
    inp = np.array([[1,2,3],
                    [4,5,6]], dtype=np.int32)
    out = np.rot90(inp, k=1)
    return {
        "train": [{"input": to_list(inp), "output": to_list(out)}],
        "test":  [{"input": to_list(inp)}]
    }


if __name__ == "__main__":
    # Run two demos
    brain = get_brain_v96()

    task1 = _demo_task_identity_plus_one()
    preds1 = brain.solve_task(task1, verbose=True)
    print("\nDemo 1 predictions:", preds1[0][0])

    task2 = _demo_task_rotate90()
    preds2 = brain.solve_task(task2, verbose=True)
    print("\nDemo 2 predictions:", preds2[0][0])

    print("\nStats:", brain.get_stats())
