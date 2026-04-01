"""3D antenna geometry visualiser using Plotly.

Renders an interactive, rotatable 3D view of the antenna design with
dimension annotations.  Designed to be shape-agnostic: new antenna types
(circular, fractal, etc.) are added by registering a renderer function.
"""

from __future__ import annotations

import math
from typing import Callable

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Registry – maps shape name → renderer callable
# ---------------------------------------------------------------------------
_RENDERERS: dict[str, Callable[..., go.Figure]] = {}


def register(name: str):
    """Decorator to register a renderer under *name*."""
    def _wrap(fn: Callable[..., go.Figure]):
        _RENDERERS[name] = fn
        return fn
    return _wrap


def available_shapes() -> list[str]:
    return list(_RENDERERS.keys())


def render(shape: str, dims: dict, **kw) -> go.Figure:
    """Dispatch to the registered renderer for *shape*."""
    if shape not in _RENDERERS:
        raise ValueError(f"Unknown shape '{shape}'. Available: {available_shapes()}")
    return _RENDERERS[shape](dims, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(x0, y0, z0, dx, dy, dz, color, name, opacity=0.85):
    """Return a Mesh3d trace for an axis-aligned box."""
    xs = [x0, x0+dx, x0+dx, x0,    x0, x0+dx, x0+dx, x0]
    ys = [y0, y0,    y0+dy, y0+dy,  y0, y0,    y0+dy, y0+dy]
    zs = [z0, z0,    z0,    z0,     z0+dz, z0+dz, z0+dz, z0+dz]
    i = [0,0,4,4, 0,0, 2,2, 0,0, 1,1]
    j = [1,2,5,6, 1,4, 3,7, 3,4, 2,5]
    k = [2,3,6,7, 4,5, 7,6, 4,7, 5,6]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        flatshading=True, hoverinfo="name",
    )


def _dim_line(p0, p1, label, fig):
    """Add a dimension annotation line between two 3-D points."""
    fig.add_trace(go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode="lines+text",
        line=dict(color="black", width=3),
        text=["", label], textposition="top center",
        textfont=dict(size=11), showlegend=False, hoverinfo="skip",
    ))


# ---------------------------------------------------------------------------
# Built-in renderer: rectangular patch
# ---------------------------------------------------------------------------

@register("rectangular")
def _render_rectangular(dims: dict, **_kw) -> go.Figure:
    pl = dims["patch_length_mm"]
    pw = dims["patch_width_mm"]
    sh = dims["substrate_height_mm"]
    sl = dims["substrate_length_mm"]
    sw = dims["substrate_width_mm"]
    gnd_t = 0.035  # copper thickness (mm), cosmetic

    # Centre everything on XY origin
    sx0, sy0 = -sl / 2, -sw / 2
    px0 = -pl / 2
    py0 = -pw / 2

    fig = go.Figure()

    # Ground plane
    fig.add_trace(_box(sx0, sy0, 0, sl, sw, gnd_t,
                       "#b0b0b0", "Ground plane"))
    # Substrate
    fig.add_trace(_box(sx0, sy0, gnd_t, sl, sw, sh,
                       "#2e8b57", "Substrate (FR-4)", opacity=0.45))
    # Patch
    fig.add_trace(_box(px0, py0, gnd_t + sh, pl, pw, gnd_t,
                       "#d4a017", "Patch"))

    # Dimension annotations
    z_top = gnd_t + sh + gnd_t
    _dim_line((px0, py0 - 1.5, z_top), (px0 + pl, py0 - 1.5, z_top),
              f"L={pl:.2f}", fig)
    _dim_line((px0 - 1.5, py0, z_top), (px0 - 1.5, py0 + pw, z_top),
              f"W={pw:.2f}", fig)
    _dim_line((sx0 - 2.5, sy0, 0), (sx0 - 2.5, sy0, gnd_t + sh),
              f"h={sh:.2f}", fig)

    pad = max(sl, sw) * 0.15
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[sx0 - pad, -sx0 + pad], title="mm"),
            yaxis=dict(range=[sy0 - pad, -sy0 + pad], title="mm"),
            zaxis=dict(range=[-0.5, gnd_t + sh + gnd_t + pad], title="mm"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0, y=1),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Stub renderers for future shapes – just register the name so the UI can
# list them; the body will be filled in when the geometry is defined.
# ---------------------------------------------------------------------------

@register("circular")
def _render_circular(dims: dict, **_kw) -> go.Figure:
    radius = dims.get("radius_mm", 10)
    sh = dims.get("substrate_height_mm", 1.6)
    sr = dims.get("substrate_radius_mm", radius * 1.5)
    gnd_t = 0.035
    n = 48  # polygon resolution

    angles = [2 * math.pi * i / n for i in range(n)]

    def _disk(cx, cy, z, r, color, name, opacity=0.85):
        xs = [cx] + [cx + r * math.cos(a) for a in angles]
        ys = [cy] + [cy + r * math.sin(a) for a in angles]
        zs = [z] * (n + 1)
        ii, jj, kk = [], [], []
        for t in range(1, n):
            ii.append(0); jj.append(t); kk.append(t + 1)
        ii.append(0); jj.append(n); kk.append(1)
        return go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                         color=color, opacity=opacity, name=name,
                         flatshading=True, hoverinfo="name")

    fig = go.Figure()
    fig.add_trace(_disk(0, 0, 0, sr, "#b0b0b0", "Ground plane"))
    fig.add_trace(_disk(0, 0, gnd_t + sh, radius, "#d4a017", "Patch"))

    pad = sr * 0.3
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-sr - pad, sr + pad], title="mm"),
            yaxis=dict(range=[-sr - pad, sr + pad], title="mm"),
            zaxis=dict(range=[-0.5, gnd_t + sh + gnd_t + pad], title="mm"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0, y=1), height=480,
    )
    return fig
