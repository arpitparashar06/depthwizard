"""
mesh_builder.py - height map + RGB -> textured 3D mesh (.glb / .obj)

Design notes for the demo:

  * A full-resolution grid is far too heavy. A 1024x614 image is 628k vertices
    and 1.25M triangles, which will crawl in a browser. We downsample to a
    target grid (default 320) -> ~100k tris, which loads instantly and still
    looks detailed once the photo is on it.

  * GLB is the format to use. It packs geometry AND texture into ONE binary
    file, so gr.Model3D and three.js can load it with no extra plumbing.
    OBJ needs a sidecar .mtl plus the image and will silently lose the texture
    if any of the three go missing.

  * Vertical exaggeration matters. At true 1:1 scale a city looks almost flat
    from above, because a 30 m building across a 1 km scene is 3% of the width.
    Expose it as a slider; 2-3x reads best.
"""

import os
import numpy as np
from PIL import Image
import trimesh



def _block_reduce(a, step, stat="median"):
    """Downsample by whole blocks rather than point sampling."""
    if step <= 1:
        return a
    H, W = a.shape
    h, w = H // step, W // step
    if h < 2 or w < 2:
        return a[::step, ::step]
    blocks = a[:h * step, :w * step].reshape(h, step, w, step).swapaxes(1, 2)
    flat = blocks.reshape(h, w, step * step)
    if np.isfinite(flat).all():
        return np.median(flat, axis=2) if stat == "median" else flat.max(axis=2)
    return np.nanmedian(flat, axis=2)



def _stepped_geometry(z, gsd, wall_min_m=0.25):
    """One flat quad per cell, plus a vertical wall wherever neighbours differ.

    A shared-vertex grid interpolates between cell centres, so a roof edge
    becomes a 45-degree ramp no matter how sharp the height map is. That single
    fact is most of why the render reads as hills instead of buildings. Here
    every cell gets its own four corners at its own height, and the gap between
    two cells is closed by an actual vertical face.

    Costs roughly 4x the vertices of the smooth grid, so drop target_grid when
    using it. Returns (verts, faces, uv).
    """
    # Quantise to the wall step. Two neighbours are then either exactly equal
    # (no gap, no wall needed) or differ by a full step (wall emitted), so the
    # surface has no hairline cracks to see through. Terracing at 0.1-0.2 m is
    # far below what reads on screen.
    if wall_min_m > 0:
        z = np.round(z / wall_min_m) * wall_min_m

    h, w = z.shape
    xs = np.arange(w + 1) * gsd
    ys = (h - np.arange(h + 1)) * gsd          # row 0 at the top, world Y up

    x0 = np.repeat(xs[:-1][None, :], h, 0); x1 = np.repeat(xs[1:][None, :], h, 0)
    y0 = np.repeat(ys[:-1][:, None], w, 1); y1 = np.repeat(ys[1:][:, None], w, 1)

    # corner order A(top-left) B(top-right) C(bottom-left) D(bottom-right)
    corners = np.stack([np.stack([x0, y0], -1), np.stack([x1, y0], -1),
                        np.stack([x0, y1], -1), np.stack([x1, y1], -1)], axis=2)
    zc = np.repeat(z[:, :, None], 4, axis=2)
    verts = np.concatenate([corners, zc[..., None]], axis=-1).reshape(-1, 3)

    u0 = np.repeat((np.arange(w) / w)[None, :], h, 0)
    u1 = np.repeat(((np.arange(w) + 1) / w)[None, :], h, 0)
    v0 = np.repeat((1 - np.arange(h) / h)[:, None], w, 1)
    v1 = np.repeat((1 - (np.arange(h) + 1) / h)[:, None], w, 1)
    uv = np.stack([np.stack([u0, v0], -1), np.stack([u1, v0], -1),
                   np.stack([u0, v1], -1), np.stack([u1, v1], -1)],
                  axis=2).reshape(-1, 2)

    base = (np.arange(h * w) * 4).reshape(h, w)
    A, B, C, D = base, base + 1, base + 2, base + 3
    faces = [np.column_stack([C.ravel(), D.ravel(), B.ravel()]),
             np.column_stack([C.ravel(), B.ravel(), A.ravel()])]

    vlist, flist, uvlist = [verts], faces, [uv]
    nxt = verts.shape[0]

    def wall(pa, pb, za, zb, uva, uvb):
        """Vertical quad from edge (pa,pb) spanning za..zb."""
        n = pa.shape[0]
        vs = np.concatenate([np.column_stack([pa, za]), np.column_stack([pb, za]),
                             np.column_stack([pa, zb]), np.column_stack([pb, zb])], axis=0)
        # order: 0=a@za 1=b@za 2=a@zb 3=b@zb
        i = nxt + np.arange(n)
        i0, i1, i2, i3 = i, i + n, i + 2 * n, i + 3 * n
        fs = [np.column_stack([i0, i1, i3]), np.column_stack([i0, i3, i2])]
        uvs = np.concatenate([uva, uvb, uva, uvb], axis=0)
        return vs, fs, uvs

    eps = float(wall_min_m) * 0.5
    # --- walls between horizontal neighbours (shared edge at x = xs[j+1]) ---
    d = z[:, 1:] - z[:, :-1]
    m = np.abs(d) > eps
    if m.any():
        ii, jj = np.nonzero(m)
        xe = xs[jj + 1]
        pa = np.column_stack([xe, ys[ii]]); pb = np.column_stack([xe, ys[ii + 1]])
        za, zb = z[ii, jj], z[ii, jj + 1]
        # take UV from the higher cell so the wall inherits the roof edge
        hi = np.where(zb > za, jj + 1, jj)
        uu = (hi + 0.5) / w
        uva = np.column_stack([uu, 1 - ii / h])
        uvb = np.column_stack([uu, 1 - (ii + 1) / h])
        vs, fs, uvs = wall(pa, pb, za, zb, uva, uvb)
        vlist.append(vs); flist += fs; uvlist.append(uvs); nxt += vs.shape[0]

    # --- walls between vertical neighbours (shared edge at y = ys[i+1]) ---
    d = z[1:, :] - z[:-1, :]
    m = np.abs(d) > eps
    if m.any():
        ii, jj = np.nonzero(m)
        ye = ys[ii + 1]
        pa = np.column_stack([xs[jj], ye]); pb = np.column_stack([xs[jj + 1], ye])
        za, zb = z[ii, jj], z[ii + 1, jj]
        hi = np.where(zb > za, ii + 1, ii)
        vv = 1 - (hi + 0.5) / h
        uva = np.column_stack([jj / w, vv])
        uvb = np.column_stack([(jj + 1) / w, vv])
        vs, fs, uvs = wall(pa, pb, za, zb, uva, uvb)
        vlist.append(vs); flist += fs; uvlist.append(uvs); nxt += vs.shape[0]

    return (np.concatenate(vlist, 0), np.concatenate(flist, 0),
            np.concatenate(uvlist, 0))


def build_mesh(height, rgb, px_size_m=None, target_grid=320,
               z_exaggeration=2.0, relative_height_m=60.0, style="stepped",
               wall_min_m=0.25, is_relative=None):
    """
    height : 2D array. Metres (absolute mode) or 0..1 (relative mode).
    rgb    : HxWx3 uint8, draped as the texture.
    px_size_m : ground sampling distance. If None, 1 unit = 1 pixel.
    relative_height_m : in relative mode, what 0..1 should map to in metres.
                        Purely for a plausible-looking demo.

    Returns a trimesh.Trimesh with UV texture.
    """
    H, W = height.shape

    # ---- 1. downsample to a manageable grid ----
    # z[::step, ::step] point-samples, which aliases badly: a sharp roofline is
    # 2-3 px wide after refinement, so stride sampling either lands on it or
    # misses it entirely and the edge turns into a ramp. Block median keeps the
    # step and is robust to the odd spike.
    step = max(1, int(round(max(H, W) / target_grid)))
    z = _block_reduce(np.asarray(height, np.float64), step)
    h, w = z.shape

    # fill holes so the mesh has no spikes
    if not np.isfinite(z).all():
        z = np.where(np.isfinite(z), z, np.nanmedian(z[np.isfinite(z)]))

    # ---- 2. put Z into metres ----
    # Sniffing the range for "is this 0..1?" is fragile: any upstream step that
    # nudges the max to 1.0001 skips the x60 stretch, and the whole scene
    # flattens to 1 m of relief with no error anywhere. Callers that know the
    # mode should pass is_relative explicitly; the sniff is only a fallback.
    if is_relative is None:
        is_relative = (np.nanmax(height) <= 1.0 + 1e-6
                       and np.nanmin(height) >= -1e-6)
    if is_relative:
        z = z * relative_height_m           # relative mode: stretch to a sane range
    z_base = float(z.min())
    z = (z - z_base) * z_exaggeration       # zero the base, then exaggerate

    # Anything reading a height off this mesh must undo both steps, or it reports
    # exaggerated metres. Carried in metadata so the viewer can invert it:
    #     true_metres = mesh_z / z_exaggeration + base_m

    # ---- 3. X/Y in metres so slopes are geometrically meaningful ----
    gsd = (px_size_m or 1.0) * step

    if style == "stepped":
        verts, faces, uv = _stepped_geometry(z, gsd, wall_min_m)
        return _finish(verts, faces, uv, rgb, z_exaggeration, z_base, gsd, step)

    xs = np.arange(w) * gsd
    ys = np.arange(h) * gsd
    xx, yy = np.meshgrid(xs, ys)

    # Y is flipped: image rows go down, world Y goes up
    verts = np.column_stack([xx.ravel(), (ys.max() - yy).ravel(), z.ravel()])

    # ---- 4. two triangles per grid cell ----
    idx = np.arange(h * w).reshape(h, w)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.vstack([np.column_stack([tl, bl, tr]),
                       np.column_stack([tr, bl, br])])

    # ---- 5. UVs: straight linear map, so the photo lands exactly on the terrain ----
    u = (xx / max(xs.max(), 1e-9)).ravel()
    v = 1.0 - (yy / max(ys.max(), 1e-9)).ravel()     # flip V for image convention
    uv = np.column_stack([u, v])

    return _finish(verts, faces, uv, rgb, z_exaggeration, z_base, gsd, step)


def _finish(verts, faces, uv, rgb, z_exaggeration, z_base, gsd, step,
            max_texture=4096):
    """Texture, material and metadata. Shared by both mesh styles."""
    tex = Image.fromarray(np.asarray(rgb, np.uint8)).convert("RGB")
    if max(tex.size) > max_texture:
        tex.thumbnail((max_texture, max_texture), Image.LANCZOS)

    # doubleSided: the wall quads' winding depends on which neighbour is higher,
    # and a back-facing wall is an invisible wall. Cheaper to let the renderer
    # draw both sides than to get every case right.
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, doubleSided=True,
        metallicFactor=0.0, roughnessFactor=0.95)

    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, image=tex, material=mat),
        process=False)          # process=True would weld vertices and break the UVs
    mesh.metadata.update(z_exaggeration=float(z_exaggeration), base_m=z_base,
                         gsd_m=float(gsd), downsample_step=int(step))
    return mesh


def export_mesh(mesh, out_path):
    """GLB by default. Returns the path actually written."""
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".glb", ".obj", ".ply"):
        out_path += ".glb"
        ext = ".glb"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    mesh.export(out_path)
    v = np.asarray(mesh.vertices)
    relief = float(v[:, 2].max() - v[:, 2].min())
    ground = float(max(v[:, 0].max() - v[:, 0].min(),
                       v[:, 1].max() - v[:, 1].min()))
    pct = 100.0 * relief / max(ground, 1e-9)
    print(f"[mesh] {len(mesh.vertices)} verts / {len(mesh.faces)} tris "
          f"-> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"[mesh] relief {relief:.1f} over {ground:.0f} of ground ({pct:.1f}%)")
    if pct < 1.0:
        # a silently flat mesh cost a lot of debugging once; say so out loud
        print("[mesh] WARNING: under 1% relief - this will render as a flat "
              "plane. Check is_relative and the height units.")
    return out_path


def height_to_mesh_file(height, rgb, out_path="outputs/terrain.glb", **kw):
    """One-call convenience wrapper for the app."""
    return export_mesh(build_mesh(height, rgb, **kw), out_path)


def slope_map(height, px_size_m=1.0):
    """Slope in degrees. The PS asks for slope analysis, and this is the whole
    implementation: gradient magnitude in metres per metre, then arctan."""
    gy, gx = np.gradient(np.asarray(height, float), px_size_m, px_size_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


if __name__ == "__main__":
    # smoke test with a synthetic hill + blocks
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W]
    z = 30 + 12 * np.sin(xx / 40.) + 9 * np.cos(yy / 35.)
    z[60:100, 60:110] += 25
    z[150:190, 140:200] += 40
    rgb = np.dstack([(z - z.min()) / (z.max() - z.min()) * 255] * 3).astype(np.uint8)
    m = build_mesh(z, rgb, px_size_m=0.5)
    export_mesh(m, "outputs/test_terrain.glb")
    print("slope max:", slope_map(z, 0.5).max().round(1), "deg")


# ---------------------------------------------------------------------------
def build_city(dsm, ndsm, rgb, px_size_m=None, target_grid=256,
               z_exaggeration=1.5, style="stepped", wall_min_m=0.25,
               is_relative=False, facade_tile_m=(12.0, 24.0), **kw):
    """Ground surface + extruded buildings, as one glTF scene.

    The heightfield alone is why the render reads as terrain: a tower is a bump
    and its sides are stretched roof pixels. Here the buildings are lifted out
    of the surface and put back as separate prisms with their own material, so
    walls are vertical, flat, and textured with something that looks like a
    building instead of a smear of roof.

    Three geometries, three materials:
        ground - the heightfield with buildings removed, satellite photo
        roofs  - flat caps, also the satellite photo, so they stay real
        walls  - procedural facade; a nadir image never saw one

    Returns (scene, info).
    """
    import trimesh
    import buildings as B

    px = float(px_size_m or 1.0)
    foot = B.extract_footprints(ndsm, rgb, px_size_m=px, **kw)
    ground_z = B.flatten_ground(dsm, foot, px_size_m=px)

    ground = build_mesh(ground_z, rgb, px_size_m=px_size_m,
                        target_grid=target_grid, z_exaggeration=z_exaggeration,
                        style=style, wall_min_m=wall_min_m,
                        is_relative=is_relative)
    md = dict(ground.metadata)
    step = md["downsample_step"]
    gsd = md["gsd_m"]
    # must match _stepped_geometry's ys exactly or the buildings sit off-grid
    y_top = (ground_z.shape[0] // step) * gsd

    rv, rf, ruv, wv, wf, wuv = B.building_meshes(
        foot, ground_z, px, y_top, z_base=md["base_m"],
        z_exaggeration=z_exaggeration, image_shape=ground_z.shape,
        facade_tile_m=facade_tile_m)

    scene = trimesh.Scene()
    scene.add_geometry(ground, geom_name="ground")

    tex = Image.fromarray(np.asarray(rgb, np.uint8)).convert("RGB")
    if max(tex.size) > 4096:
        tex.thumbnail((4096, 4096), Image.LANCZOS)

    if len(rf):
        roofs = trimesh.Trimesh(
            vertices=rv, faces=rf, process=False,
            visual=trimesh.visual.TextureVisuals(
                uv=ruv, image=tex,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex, doubleSided=True,
                    metallicFactor=0.0, roughnessFactor=0.9)))
        scene.add_geometry(roofs, geom_name="roofs")

    if len(wf):
        fac = B.facade_texture()
        walls = trimesh.Trimesh(
            vertices=wv, faces=wf, process=False,
            visual=trimesh.visual.TextureVisuals(
                uv=wuv, image=fac,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=fac, doubleSided=True,
                    metallicFactor=0.0, roughnessFactor=0.85)))
        scene.add_geometry(walls, geom_name="walls")

    info = dict(B.summarise(foot), ground_tris=int(len(ground.faces)),
                roof_tris=int(len(rf)), wall_tris=int(len(wf)))
    print(f"[city] {info['n']} buildings | ground {info['ground_tris']:,} tris, "
          f"roofs {info['roof_tris']:,}, walls {info['wall_tris']:,}")
    scene.metadata.update(md)
    return scene, info
