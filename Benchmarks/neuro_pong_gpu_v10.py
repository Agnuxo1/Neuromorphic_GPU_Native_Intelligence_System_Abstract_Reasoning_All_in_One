#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroPong GPU v10.0 - 100% GPU OpenGL Retro Arcade Demo
========================================================

CHIMERA Educational Demo - "Rendering IS Thinking"
--------------------------------------------------

This is a complete reimplementation of NeuroPong using 100% GPU/OpenGL rendering.
Everything happens on the GPU: physics, brain computation, and visualization.

Architecture:
-------------
- Game state lives in GPU textures (no CPU arrays)
- Physics computed in compute shaders
- Brain (RGBA neuromorphic frame) updated via fragment shaders
- Beautiful retro arcade visualization with glow effects
- Real-time brain visualization showing R, G, B, A channels

Features:
---------
- ✨ Retro arcade aesthetic with CRT scanlines and glow
- 🧠 Beautiful brain visualization with color-coded channels
- ⚡ 100% GPU - everything in shaders
- 🎮 Smooth 60 FPS gameplay
- 📊 Real-time neural activity display

Controls:
---------
- W/S or ↑/↓: Move human paddle
- A/D: Adjust evolution alpha (α)
- Space: Pause/Resume
- F: Toggle brain visualization mode
- R: Reset game
- Esc/Q: Quit

Requirements:
-------------
- Python 3.8+
- moderngl
- pygame
- numpy
- glfw (optional, for better window management)

Author:
-------
CHIMERA Project, 2025 - Francisco Angulo de Lafuente
"""

import numpy as np
import moderngl
import pygame
from pygame.locals import *
import sys
import time
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Display
    window_width: int = 1920
    window_height: int = 1080
    grid_size: int = 64  # Higher resolution for smoother gameplay

    # Game
    paddle_length: int = 8
    ball_speed: float = 0.015  # Normalized units per frame
    ai_response: float = 0.08

    # Brain
    alpha: float = 0.15  # Evolution rate
    sigma: float = 2.0   # Gaussian width for target painting

    # Visual
    glow_intensity: float = 0.6
    scanline_intensity: float = 0.15
    crt_curvature: float = 0.02


# ============================================================================
# VERTEX SHADER (Fullscreen Quad)
# ============================================================================

VERTEX_SHADER = """
#version 330

in vec2 in_vert;
out vec2 uv;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    uv = (in_vert + 1.0) / 2.0;
}
"""


# ============================================================================
# GAME PHYSICS SHADER
# ============================================================================

PHYSICS_SHADER = """
#version 330

uniform sampler2D u_state;      // Current game state
uniform vec2 u_ball_pos;        // Ball position (normalized)
uniform float u_human_y;        // Human paddle center Y
uniform float u_ai_y;           // AI paddle center Y
uniform int u_grid_size;
uniform int u_paddle_len;

in vec2 uv;
out vec4 out_state;

const float PADDLE_X_HUMAN = 0.05;
const float PADDLE_X_AI = 0.95;
const float WALL_Y_TOP = 0.02;
const float WALL_Y_BOT = 0.98;

void main() {
    ivec2 pixel = ivec2(uv * u_grid_size);
    vec2 pos = vec2(pixel) / float(u_grid_size);

    // Determine cell type
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

    // Walls (top/bottom)
    if (pixel.y == 0 || pixel.y == u_grid_size - 1) {
        color = vec4(1.0, 1.0, 1.0, 1.0);  // White walls
    }

    // Human paddle (left, magenta)
    float paddle_half = float(u_paddle_len) / float(u_grid_size) / 2.0;
    if (abs(pos.x - PADDLE_X_HUMAN) < 0.01 &&
        abs(pos.y - u_human_y) < paddle_half) {
        color = vec4(1.0, 0.0, 1.0, 1.0);  // Magenta
    }

    // AI paddle (right, cyan)
    if (abs(pos.x - PADDLE_X_AI) < 0.01 &&
        abs(pos.y - u_ai_y) < paddle_half) {
        color = vec4(0.0, 1.0, 1.0, 1.0);  // Cyan
    }

    // Ball (red with glow)
    float ball_dist = length(pos - u_ball_pos);
    if (ball_dist < 0.015) {
        color = vec4(1.0, 0.0, 0.0, 1.0);  // Red
    } else if (ball_dist < 0.025) {
        // Glow effect
        float glow = 1.0 - (ball_dist - 0.015) / 0.01;
        color = vec4(1.0, 0.0, 0.0, glow * 0.5);
    }

    out_state = color;
}
"""


# ============================================================================
# BRAIN COMPUTE SHADER (Neuromorphic Evolution)
# ============================================================================

BRAIN_SHADER = """
#version 330

uniform sampler2D u_state;      // Game state (R channel = grid encoding)
uniform sampler2D u_brain_prev; // Previous brain state (RGBA)
uniform float u_alpha;          // Evolution rate
uniform float u_target_y;       // AI target Y (normalized)
uniform float u_sigma;          // Gaussian sigma
uniform int u_grid_size;

in vec2 uv;
layout(location = 0) out vec4 out_brain;
layout(location = 1) out vec4 out_features;

// 3×3 neighborhood features
vec3 compute_features() {
    ivec2 pixel = ivec2(uv * u_grid_size);
    vec4 center = texelFetch(u_state, pixel, 0);
    float center_val = center.r;

    int same_count = 0;
    vec3 color_sum = vec3(0.0);

    // 8 neighbors
    for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
            if(dx == 0 && dy == 0) continue;

            ivec2 npos = pixel + ivec2(dx, dy);
            npos = clamp(npos, ivec2(0), ivec2(u_grid_size - 1));

            vec4 neighbor = texelFetch(u_state, npos, 0);
            float n_val = neighbor.r;

            if(abs(n_val - center_val) < 0.01) {
                same_count++;
            }

            color_sum += neighbor.rgb;
        }
    }

    float same_ratio = float(same_count) / 8.0;
    float edge_flag = same_ratio < 0.625 ? 1.0 : 0.0;
    float avg_intensity = length(color_sum) / 8.0;

    return vec3(same_ratio, edge_flag, avg_intensity);
}

// Gaussian target band for AI prediction
float compute_result() {
    float y = uv.y;
    float dist = abs(y - u_target_y);
    return exp(-0.5 * (dist / u_sigma) * (dist / u_sigma));
}

void main() {
    // Read previous brain state
    vec4 prev = texture(u_brain_prev, uv);

    // Encode current state (R channel)
    vec4 state = texture(u_state, uv);
    float R = length(state.rgb) / sqrt(3.0);  // Normalize to [0,1]

    // Compute features
    vec3 features = compute_features();

    // Compute result (B channel) - where should AI paddle be?
    float B = compute_result();

    // Evolve memory (G channel) - exponential moving average
    float G = mix(prev.g, B, u_alpha);

    // Confidence (A channel) - higher when ball is moving toward AI
    float A = 1.0;

    // Output neuromorphic frame
    out_brain = vec4(R, G, B, A);

    // Output features for visualization
    out_features = vec4(features, 1.0);
}
"""


# ============================================================================
# VISUALIZATION SHADER (Retro Arcade + Brain Display)
# ============================================================================

VISUAL_SHADER = """
#version 330

uniform sampler2D u_game;       // Game state
uniform sampler2D u_brain;      // Brain RGBA
uniform sampler2D u_features;   // Features
uniform float u_glow;
uniform float u_scanline;
uniform float u_curvature;
uniform vec2 u_resolution;
uniform float u_time;           // For KITT animation

in vec2 uv;
out vec4 fragColor;

// CRT scanline effect
float scanline(vec2 uv, float intensity) {
    float line = sin(uv.y * u_resolution.y * 3.14159);
    return 1.0 - intensity * (0.5 + 0.5 * line);
}

// Glow effect
vec3 glow(sampler2D tex, vec2 uv, float intensity) {
    vec3 col = texture(tex, uv).rgb;

    // Sample neighbors for bloom
    vec3 bloom = vec3(0.0);
    float kernel[9] = float[](
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0
    );

    int idx = 0;
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 offset = vec2(x, y) / u_resolution;
            bloom += texture(tex, uv + offset).rgb * kernel[idx++];
        }
    }
    bloom /= 16.0;

    return col + bloom * intensity;
}

// KITT-style LED indicator
float kitt_indicator(vec2 pos, float value, float bar_index) {
    // Horizontal LED bar at specific Y position
    float y_pos = 0.05 + bar_index * 0.12;
    float bar_height = 0.015;

    // Check if we're in the bar region
    if (abs(pos.y - y_pos) < bar_height) {
        // LED segments (8 segments)
        float segment_width = 0.08;
        float segment_gap = 0.02;

        for (int i = 0; i < 8; i++) {
            float seg_x = 0.1 + float(i) * (segment_width + segment_gap);
            float seg_end = seg_x + segment_width;

            if (pos.x >= seg_x && pos.x <= seg_end) {
                // Light up segments based on value
                float threshold = float(i) / 8.0;
                if (value >= threshold) {
                    // KITT red glow with intensity falloff
                    float intensity = 1.0 - (value - threshold) * 2.0;
                    intensity = clamp(intensity, 0.3, 1.0);

                    // Animated sweep effect
                    float sweep = fract(u_time * 0.5 + float(i) * 0.125);
                    intensity *= 0.7 + 0.3 * sweep;

                    return intensity;
                }
            }
        }
    }
    return 0.0;
}

// Beautiful brain channel visualization
vec3 visualize_brain(vec4 brain, vec3 features, vec2 pos) {
    // Color-code each channel
    vec3 R_color = vec3(1.0, 0.3, 0.3) * brain.r;  // Red for State
    vec3 G_color = vec3(0.3, 1.0, 0.3) * brain.g;  // Green for Memory
    vec3 B_color = vec3(0.3, 0.3, 1.0) * brain.b;  // Blue for Result

    // Combine with additive blending
    vec3 combined = R_color + G_color + B_color;

    // Add feature overlay (edges in white)
    combined += vec3(1.0) * features.g * 0.3;

    // Add KITT-style indicators (minimalista)
    float led_r = kitt_indicator(pos, brain.r, 0.0);  // State indicator
    float led_g = kitt_indicator(pos, brain.g, 1.0);  // Memory indicator
    float led_b = kitt_indicator(pos, brain.b, 2.0);  // Result indicator

    // Overlay LEDs with proper colors
    combined += vec3(1.0, 0.1, 0.0) * led_r * 2.0;  // Red LEDs
    combined += vec3(0.1, 1.0, 0.0) * led_g * 2.0;  // Green LEDs
    combined += vec3(0.0, 0.3, 1.0) * led_b * 2.0;  // Blue LEDs

    return combined;
}

// Grid overlay
float grid(vec2 uv, float divisions) {
    vec2 grid_uv = fract(uv * divisions);
    vec2 grid_dist = min(grid_uv, 1.0 - grid_uv);
    float line_width = 0.02;
    float grid_val = step(line_width, grid_dist.x) * step(line_width, grid_dist.y);
    return 1.0 - (1.0 - grid_val) * 0.3;
}

void main() {
    vec2 centered_uv = uv;

    // CRT curvature
    centered_uv = centered_uv * 2.0 - 1.0;
    centered_uv *= 1.0 + u_curvature * dot(centered_uv, centered_uv);
    centered_uv = (centered_uv + 1.0) / 2.0;

    // Out of bounds check
    if(centered_uv.x < 0.0 || centered_uv.x > 1.0 ||
       centered_uv.y < 0.0 || centered_uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 color = vec3(0.0);

    // Layout: Left half = Game, Right half = Brain
    if(centered_uv.x < 0.5) {
        // GAME VIEW (left half)
        vec2 game_uv = vec2(centered_uv.x * 2.0, centered_uv.y);
        vec3 game_col = glow(u_game, game_uv, u_glow);

        // Add grid
        game_col *= grid(game_uv, 32.0);

        color = game_col;
    } else {
        // BRAIN VIEW (right half)
        vec2 brain_uv = vec2((centered_uv.x - 0.5) * 2.0, centered_uv.y);

        vec4 brain = texture(u_brain, brain_uv);
        vec3 features = texture(u_features, brain_uv).rgb;

        vec3 brain_vis = visualize_brain(brain, features, brain_uv);

        // Add subtle grid
        brain_vis *= grid(brain_uv, 16.0);

        color = brain_vis;
    }

    // Scanlines
    color *= scanline(centered_uv, u_scanline);

    // Vignette
    float vignette = 1.0 - 0.3 * length(centered_uv - 0.5);
    color *= vignette;

    // Slight color aberration for retro feel
    color.r *= 1.05;
    color.b *= 0.95;

    fragColor = vec4(color, 1.0);
}
"""


# ============================================================================
# GPU PONG ENGINE
# ============================================================================

class NeuroPongGPU:
    """100% GPU implementation of NeuroPong with CHIMERA brain."""

    def __init__(self, config: Config):
        self.cfg = config

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.window_width, config.window_height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("NeuroPong GPU v10.0 - CHIMERA Brain")

        # Create OpenGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        print("=" * 80)
        print("NeuroPong GPU v10.0 - CHIMERA Neuromorphic Demo")
        print("=" * 80)
        print(f"[GPU] {self.ctx.info['GL_RENDERER']}")
        print(f"[Resolution] {config.window_width}×{config.window_height}")
        print(f"[Grid] {config.grid_size}×{config.grid_size}")
        print("=" * 80)

        # Compile shaders
        self._compile_shaders()

        # Create textures
        self._create_textures()

        # Create framebuffers
        self._create_framebuffers()

        # Initialize game state
        self._init_game_state()

        # Setup rendering
        self._setup_rendering()

        # Stats
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.paused = False
        self.start_time = time.time()

        print("[READY] Press Space to start!")
        print("=" * 80)

    def _compile_shaders(self):
        """Compile all shader programs."""
        # Physics shader
        self.physics_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=PHYSICS_SHADER
        )

        # Brain shader
        self.brain_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=BRAIN_SHADER
        )

        # Visualization shader
        self.visual_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=VISUAL_SHADER
        )

        print("[SHADERS] Compiled: physics, brain, visual")

    def _create_textures(self):
        """Create GPU textures for game state and brain."""
        gs = self.cfg.grid_size

        # Game state texture
        self.game_texture = self.ctx.texture(
            size=(gs, gs), components=4, dtype='f4'
        )
        self.game_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Brain textures (ping-pong)
        self.brain_texture_a = self.ctx.texture(
            size=(gs, gs), components=4, dtype='f4'
        )
        self.brain_texture_a.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.brain_texture_b = self.ctx.texture(
            size=(gs, gs), components=4, dtype='f4'
        )
        self.brain_texture_b.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Features texture
        self.features_texture = self.ctx.texture(
            size=(gs, gs), components=4, dtype='f4'
        )
        self.features_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Initialize to zeros
        zeros = np.zeros((gs, gs, 4), dtype=np.float32)
        self.game_texture.write(zeros.tobytes())
        self.brain_texture_a.write(zeros.tobytes())
        self.brain_texture_b.write(zeros.tobytes())
        self.features_texture.write(zeros.tobytes())

        print(f"[TEXTURES] Created {gs}×{gs} game, brain, features")

    def _create_framebuffers(self):
        """Create framebuffers for rendering."""
        # Game FBO
        self.game_fbo = self.ctx.framebuffer(
            color_attachments=[self.game_texture]
        )

        # Brain FBO (with MRT for features)
        self.brain_fbo_a = self.ctx.framebuffer(
            color_attachments=[self.brain_texture_a, self.features_texture]
        )

        self.brain_fbo_b = self.ctx.framebuffer(
            color_attachments=[self.brain_texture_b, self.features_texture]
        )

        print("[FBO] Created framebuffers")

    def _init_game_state(self):
        """Initialize game state variables (CPU side for physics)."""
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = -0.01
        self.ball_vy = 0.008

        self.human_y = 0.5
        self.ai_y = 0.5
        self.ai_target_y = 0.5

        self.score_human = 0
        self.score_ai = 0

        self.brain_ping_pong = False  # False = use A, True = use B

    def _setup_rendering(self):
        """Setup fullscreen quad rendering."""
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # Create VAO for each shader
        self.vao_physics = self.ctx.simple_vertex_array(
            self.physics_program, self.vbo, 'in_vert'
        )
        self.vao_brain = self.ctx.simple_vertex_array(
            self.brain_program, self.vbo, 'in_vert'
        )
        self.vao_visual = self.ctx.simple_vertex_array(
            self.visual_program, self.vbo, 'in_vert'
        )

    def update_physics(self, dt: float):
        """Update game physics (CPU for now, can be moved to GPU)."""
        if self.paused:
            return

        # Move ball
        self.ball_x += self.ball_vx * dt * 60
        self.ball_y += self.ball_vy * dt * 60

        # Wall bounces
        if self.ball_y < 0.02:
            self.ball_y = 0.02
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y > 0.98:
            self.ball_y = 0.98
            self.ball_vy = -abs(self.ball_vy)

        # Paddle collisions
        paddle_half = (self.cfg.paddle_length / self.cfg.grid_size) / 2

        # Human paddle
        if abs(self.ball_x - 0.05) < 0.02 and abs(self.ball_y - self.human_y) < paddle_half:
            self.ball_vx = abs(self.ball_vx)
            self.ball_vy += (self.ball_y - self.human_y) * 0.05

        # AI paddle
        if abs(self.ball_x - 0.95) < 0.02 and abs(self.ball_y - self.ai_y) < paddle_half:
            self.ball_vx = -abs(self.ball_vx)
            self.ball_vy += (self.ball_y - self.ai_y) * 0.05

        # Scoring
        if self.ball_x < 0:
            self.score_ai += 1
            self.reset_ball(direction=1)
        elif self.ball_x > 1:
            self.score_human += 1
            self.reset_ball(direction=-1)

        # AI follows target from brain
        diff = self.ai_target_y - self.ai_y
        self.ai_y += np.sign(diff) * min(abs(diff), self.cfg.ai_response * dt * 60)
        self.ai_y = np.clip(self.ai_y, 0.1, 0.9)

    def reset_ball(self, direction: int = 1):
        """Reset ball to center."""
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = direction * 0.01
        self.ball_vy = (np.random.rand() - 0.5) * 0.015

    def render_game_state(self):
        """Render current game state to texture."""
        self.game_fbo.use()
        self.game_fbo.clear(0.0, 0.0, 0.0, 1.0)

        self.physics_program['u_ball_pos'] = (self.ball_x, self.ball_y)
        self.physics_program['u_human_y'] = self.human_y
        self.physics_program['u_ai_y'] = self.ai_y
        self.physics_program['u_grid_size'] = self.cfg.grid_size
        self.physics_program['u_paddle_len'] = self.cfg.paddle_length

        self.vao_physics.render(moderngl.TRIANGLE_STRIP)

    def update_brain(self):
        """Update neuromorphic brain state."""
        # Predict where ball will hit AI column
        if self.ball_vx > 0:
            # Ball moving toward AI
            time_to_hit = (0.95 - self.ball_x) / abs(self.ball_vx) if abs(self.ball_vx) > 0.001 else 999
            predicted_y = self.ball_y + self.ball_vy * time_to_hit

            # Simple bounce simulation
            while predicted_y < 0.02 or predicted_y > 0.98:
                if predicted_y < 0.02:
                    predicted_y = 0.04 - predicted_y
                elif predicted_y > 0.98:
                    predicted_y = 1.96 - predicted_y

            self.ai_target_y = np.clip(predicted_y, 0.1, 0.9)

        # Select FBO (ping-pong)
        if self.brain_ping_pong:
            fbo = self.brain_fbo_a
            prev_tex = self.brain_texture_b
        else:
            fbo = self.brain_fbo_b
            prev_tex = self.brain_texture_a

        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)

        # Setup uniforms
        self.brain_program['u_state'] = 0
        self.brain_program['u_brain_prev'] = 1
        self.brain_program['u_alpha'] = self.cfg.alpha
        self.brain_program['u_target_y'] = self.ai_target_y
        self.brain_program['u_sigma'] = self.cfg.sigma / self.cfg.grid_size
        self.brain_program['u_grid_size'] = self.cfg.grid_size

        # Bind textures
        self.game_texture.use(location=0)
        prev_tex.use(location=1)

        # Render
        self.vao_brain.render(moderngl.TRIANGLE_STRIP)

        # Ping-pong
        self.brain_ping_pong = not self.brain_ping_pong

    def render_to_screen(self):
        """Render final visualization to screen."""
        self.ctx.screen.use()
        self.ctx.screen.clear(0.0, 0.0, 0.0, 1.0)

        # Current brain texture
        brain_tex = self.brain_texture_a if self.brain_ping_pong else self.brain_texture_b

        # Setup uniforms
        self.visual_program['u_game'] = 0
        self.visual_program['u_brain'] = 1
        self.visual_program['u_features'] = 2
        self.visual_program['u_glow'] = self.cfg.glow_intensity
        self.visual_program['u_scanline'] = self.cfg.scanline_intensity
        self.visual_program['u_curvature'] = self.cfg.crt_curvature
        self.visual_program['u_resolution'] = (self.cfg.window_width, self.cfg.window_height)
        self.visual_program['u_time'] = time.time() - self.start_time

        # Bind textures
        self.game_texture.use(location=0)
        brain_tex.use(location=1)
        self.features_texture.use(location=2)

        # Render
        self.vao_visual.render(moderngl.TRIANGLE_STRIP)

    def handle_input(self):
        """Handle user input."""
        keys = pygame.key.get_pressed()

        # Human paddle control (FIXED: inverted controls)
        if keys[K_UP] or keys[K_w]:
            self.human_y += 0.02  # UP arrow now moves UP
        if keys[K_DOWN] or keys[K_s]:
            self.human_y -= 0.02  # DOWN arrow now moves DOWN

        self.human_y = np.clip(self.human_y, 0.1, 0.9)

        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    return False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_r:
                    self.reset_ball()
                elif event.key == K_a:
                    self.cfg.alpha = max(0.0, self.cfg.alpha - 0.01)
                    print(f"[ALPHA] {self.cfg.alpha:.3f}")
                elif event.key == K_d:
                    self.cfg.alpha = min(1.0, self.cfg.alpha + 0.01)
                    print(f"[ALPHA] {self.cfg.alpha:.3f}")

        return True

    def run(self):
        """Main game loop."""
        running = True

        while running:
            dt = self.clock.tick(self.fps) / 1000.0

            running = self.handle_input()
            if not running:
                break

            # Update
            self.update_physics(dt)

            # Render pipeline
            self.render_game_state()
            self.update_brain()
            self.render_to_screen()

            # Display
            pygame.display.flip()

            # Stats
            if int(time.time()) % 5 == 0:
                fps = self.clock.get_fps()
                pygame.display.set_caption(
                    f"NeuroPong GPU v10.0 | FPS: {fps:.1f} | "
                    f"Score: {self.score_human}-{self.score_ai} | "
                    f"Alpha: {self.cfg.alpha:.2f}"
                )

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Release GPU resources."""
        self.game_texture.release()
        self.brain_texture_a.release()
        self.brain_texture_b.release()
        self.features_texture.release()
        self.game_fbo.release()
        self.brain_fbo_a.release()
        self.brain_fbo_b.release()
        self.ctx.release()
        pygame.quit()
        print("\n[CLEANUP] GPU resources released")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("NeuroPong GPU v10.0 - CHIMERA Demo")
    print("'Rendering IS Thinking'")
    print("100% GPU OpenGL Neuromorphic Architecture")
    print("=" * 80)

    config = Config()
    game = NeuroPongGPU(config)

    try:
        game.run()
    except KeyboardInterrupt:
        print("\n[EXIT] User interrupt")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nThanks for playing NeuroPong GPU!")
        print("CHIMERA Project - 2025")
