"""Make MetaDrive's 3D renderer work on macOS (Apple GPU, Metal-backed OpenGL).

Root cause (diagnosed with Panda3D shader logging): macOS gives Panda3D an OpenGL 2.1 / GLSL 1.20
context by default. MetaDrive's terrain shader (which also draws the asphalt and lane markings) is
`#version 150`, so on that context the road shows up as a blank white plane. Requesting a core
profile (OpenGL 4.1) fixes the version, but Apple's strict GLSL compiler then rejects the legacy
`texture2D()` call that MetaDrive's terrain shader and simplepbr's PBR/tonemap shaders still use.

This module, applied BEFORE the first rendered MetaDrive env is created and without editing the
installed packages:
  * requests a core-profile context (gl-version 3 2 -> OpenGL 4.1 on macOS),
  * lowers MSAA (8x window / 16x buffer are unsupported -> "Could not open window"),
  * uses simplepbr's 330 shader variant and the standard (#version 150) skybox shaders,
  * rewrites `texture2D(` -> `texture(` in the terrain and simplepbr shader sources as they are loaded.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def apply(msaa: int = 4, shader_log: bool = False) -> None:
    if sys.platform != "darwin":
        return
    from panda3d.core import loadPrcFileData

    import metadrive.engine.core.engine_core as ec  # import first: it sets multisamples 8 at import time
    import metadrive.engine.core.sky_box as sky
    import metadrive.engine.core.terrain as terrain
    import metadrive.third_party.simplepbr as simplepbr

    loadPrcFileData("", f"multisamples {msaa}\ngl-version 3 2")
    if shader_log:                               # show GLSL compile errors (MetaDrive silences them)
        ec._suppress_warning = lambda: None
        loadPrcFileData("", "notify-level-glgsg error")

    sky.is_mac = lambda: False                   # standard skybox shaders are #version 150

    _init = ec.init
    ec.init = lambda **kw: _init(**{**kw, "msaa_samples": msaa, "use_330": True})

    _load = simplepbr._load_shader_str
    simplepbr._load_shader_str = lambda path, defines=None: _load(path, defines).replace("texture2D(", "texture(")

    patched_dir = Path(tempfile.mkdtemp(prefix="metadrive_macos_shaders_"))
    _Shader = terrain.Shader

    class _ShaderShim:
        SL_GLSL = _Shader.SL_GLSL

        @staticmethod
        def load(lang, vert, frag, *args, **kw):
            if str(frag).endswith("terrain.frag.glsl"):
                out = patched_dir / "terrain.frag.glsl"
                out.write_text(Path(frag).read_text().replace("texture2D(", "texture("))
                frag = str(out)
            return _Shader.load(lang, vert, frag, *args, **kw)

        def __getattr__(self, k):                # pragma: no cover
            return getattr(_Shader, k)

    terrain.Shader = _ShaderShim
