"""
Compatibility shims so the 2018-era Ortho4XP 1.30 code runs on current libraries.
Imported first by O4_File_Names, so every other module sees the patched behaviour.

- Shapely 2.x removed iteration over multi-part geometries (`for pol in multipolygon`),
  the `.type` attribute and `shapely.ops.cascaded_union`. This restores iteration (via
  `.geoms`), the `.type` alias and the `cascaded_union` name. It deliberately does NOT add
  `__len__` / `__getitem__`: numpy would then treat geometries as sequences and recurse.
  The few `len(geometry)` call sites in the source were changed to `len(geometry.geoms)`.
- numpy 2.x removed the `numpy.float` / `numpy.int` aliases; call sites were changed, but
  the aliases are restored here too for any code path that was missed.
- Pillow 10+ removed `Image.ANTIALIAS`; alias it to LANCZOS.
"""
import warnings

try:
    import shapely
    import shapely.ops
    from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
    try:
        iter(shapely.geometry.MultiPoint([(0, 0)]))
    except TypeError:
        BaseMultipartGeometry.__iter__ = lambda self: iter(self.geoms)
    if not hasattr(BaseMultipartGeometry, "__len__"):
        # __len__ alone is safe: numpy only treats objects with __getitem__ as sequences
        BaseMultipartGeometry.__len__ = lambda self: len(self.geoms)
    if not hasattr(BaseGeometry, 'type'):
        BaseGeometry.type = property(lambda self: self.geom_type)
    if not hasattr(shapely.ops, 'cascaded_union'):
        shapely.ops.cascaded_union = shapely.ops.unary_union
except Exception as e:  # pragma: no cover
    print("O4_Compat: shapely shim not applied:", e)

try:
    import numpy
    for _name, _alias in (('float', float), ('int', int), ('bool', bool), ('object', object)):
        if not hasattr(numpy, _name):
            setattr(numpy, _name, _alias)
except Exception as e:  # pragma: no cover
    print("O4_Compat: numpy shim not applied:", e)

try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
except Exception as e:  # pragma: no cover
    print("O4_Compat: PIL shim not applied:", e)

# pyproj keeps warning about the '+init=epsg:...' syntax Ortho4XP uses; it still works.
warnings.filterwarnings('ignore', message=".*'\\+init=<authority>:<code>' syntax is deprecated.*")
