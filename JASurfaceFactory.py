# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

from .mod_reload import reload_modules
reload_modules(locals(), __package__, ["JAG2Constants", "JAG2Panels"], [".casts", ".error_types"])  # nopep8

import bpy
import string
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, cast
from .casts import optional_cast, downcast, bpy_generic_cast, unpack_cast, matrix_getter_cast, matrix_overload_cast, vector_getter_cast, vector_overload_cast
from .error_types import ErrorMessage, NoError, ensureListIsGapless
from . import JAG2Panels
from . import JAG2Constants

class Vertex_map:
    def __init__(self, object, mesh, vertex_id, loop_id):
        self.mesh = mesh
        self.obj = object
        self.vert = vertex_id
        self.loop = loop_id
        self.position = mesh.vertices[vertex_id].co
        self.normal = mesh.vertices[vertex_id].normal
        if mesh.has_custom_normals:
            self.normal = mesh.loops[loop_id].normal
        self.tc = mesh.uv_layers.active.data[loop_id].uv
        self.hash_tuple = tuple((*self.position, *self.normal, *self.tc))

    def set_mesh(self, mesh):
        self.mesh = mesh


def getName(object: bpy.types.Object) -> str:
    if object.g2_prop_name != "":  # pyright: ignore [reportAttributeAccessIssue]
        return object.g2_prop_name  # pyright: ignore [reportAttributeAccessIssue]
    return object.name


class Surface_descriptor:
    def __init__(self, material, surface_name, parent_index, flags):
        self.current_index = 0
        self.vertex_mapping = []
        self.vertex_hashes = {}
        self.triangles = []
        self.material = material
        self.surface_name = surface_name
        self.parent_index = parent_index
        self.num_children = 0
        self.flags = flags

    # always make sure that you pack the same material in
    # one surface descriptor!
    def add_triangle(self, in_obj, in_mesh, in_triangle, SHADER_MAX_VERTEXES=1000):
        if len(self.triangles) * 3 >= 6 * SHADER_MAX_VERTEXES:
            return False

        new_triangle: List[Optional[int]] = [None, None, None]
        new_map = None

        reused_vertices = 0
        vertices = []
        for index, (tri, loo) in enumerate(zip(in_triangle.vertices,
                                               in_triangle.loops)):
            vert_map = Vertex_map(in_obj, in_mesh, tri, loo)
            vertices.append(vert_map)
            if vert_map.hash_tuple not in self.vertex_hashes:
                continue
            # vertex already in the surface
            if new_triangle[index] is None:
                new_triangle[index] = self.vertex_hashes[vert_map.hash_tuple]
                reused_vertices += 1

        if 3-reused_vertices + len(self.vertex_mapping) >= SHADER_MAX_VERTEXES:
            return False

        # add new vertices
        for id, index in enumerate(new_triangle):
            if index is None:
                new_map = vertices[id]
                self.vertex_mapping.append(new_map)
                self.vertex_hashes[new_map.hash_tuple] = self.current_index
                new_triangle[id] = self.current_index
                self.current_index += 1

        # add new triangle
        self.triangles.append(new_triangle)
        return True


class Surface_factory:
    valid = False
    status = "Unknown Error"

    def __init__(self, 
         root_object: bpy.types.Object,
         SHADER_MAX_VERTEXES=1000,
         MAX_SURFACES=32):
        surfaces = {}
        self.surface_descriptors = []
        self.num_surfaces = 0

        depsgraph = bpy.context.evaluated_depsgraph_get()

        def add_children(object: bpy.types.Object, parent_index = -1) -> Tuple[bool, ErrorMessage]:
            current_parent_index = parent_index
            object_sd = None
            
            if object.type == 'MESH':
                if not JAG2Panels.hasG2MeshProperties(object):
                    return False, ErrorMessage(f"{object.name} has no Ghoul 2 properties set! (Also, the exporter should've detected this earlier.)")
                name = getName(object)
                shader = object.g2_prop_shader # pyright: ignore [reportAttributeAccessIssue]
                flags = 0
                if object.g2_prop_off:  # pyright: ignore [reportAttributeAccessIssue]
                    flags |= JAG2Constants.SURFACEFLAG_OFF
                if object.g2_prop_tag:  # pyright: ignore [reportAttributeAccessIssue]
                    flags |= JAG2Constants.SURFACEFLAG_TAG
                
                if name in surfaces:
                    print("double encountered and skipped", name)
                    return False, ErrorMessage(
                        f"{object.name} has a surface name that is already used by another object")
                
                object_sd = Surface_descriptor(shader, name, parent_index, flags) # pyright: ignore [reportAttributeAccessIssue]
                
                current_sd = object_sd
                current_name = name
                mesh = bpy_generic_cast(bpy.types.Object, object.evaluated_get(depsgraph)).to_mesh()
                mesh.calc_loop_triangles()

                was_splitted = False

                for triangle in mesh.loop_triangles:
                    success = current_sd.add_triangle(object, mesh, triangle, SHADER_MAX_VERTEXES)

                    if not success:
                        self.surface_descriptors.append(current_sd)
                        current_parent_index = self.surface_descriptors.index(object_sd)
                        if was_splitted:
                            char_index = string.ascii_letters.index(current_name[len(current_name)-1])
                            new_char = string.ascii_letters[char_index % (len(string.ascii_letters)-1)]
                            current_name = f"{current_name[:-1]}{new_char}"
                        else:
                            current_name = f"{current_name}b"
                            was_splitted = True

                        current_sd  = Surface_descriptor(object.g2_prop_shader, current_name, current_parent_index, flags) # pyright: ignore [reportAttributeAccessIssue]
                        success = current_sd.add_triangle(object, mesh, triangle, SHADER_MAX_VERTEXES)
                        object_sd.num_children += 1
                self.surface_descriptors.append(current_sd)

            for child in object.children:
                # only meshes supported in hierarchy, I couldn't always use the parent otherwise
                if child.type != 'MESH':
                    print(
                        f"Warning: {child.name} is no mesh, neither it nor its children will be exported!")
                    continue
                if not JAG2Panels.hasG2MeshProperties(child):
                    return False, ErrorMessage(f"{child.name} has no Ghoul 2 properties set! (Also, the exporter should've detected this earlier.)")
                
                if object_sd:
                    current_parent_index = self.surface_descriptors.index(object_sd)
                success, message = add_children(child, current_parent_index)
                if not success:
                    return False, message
                if object_sd:
                    object_sd.num_children += 1
                
            return True, ErrorMessage("Nothing")

        success, message = add_children(root_object)

        if not success:
            self.valid = success
            self.status = message
            return

        self.valid = True
        self.status = "Added object(s) successfully to surface factory."
        return
