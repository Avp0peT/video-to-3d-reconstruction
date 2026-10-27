"""Blender-only presentation styling; legacy geometry/metadata checks remain exact."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import bpy

STYLES={'Inferred complete envelope':('INFERRED_COMPLETION',(1.,.38,.06,1.),False),'Observed-depth reconstruction':('OBSERVED_DEPTH',(.1,.4,.9,1.),True)}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_style(scenes):
    result=[]
    for index,source in enumerate(scenes):
        obj=bpy.data.objects.get('PLY_'+str(index+1).zfill(2))
        name,color,hidden=STYLES[source['title']]
        collection=bpy.data.collections.get(name)
        if collection is None or tuple(obj.users_collection)!=(collection,) or list(collection.objects)!=[obj]:raise ValueError('semantic layer collection differs')
        if collection.hide_viewport!=hidden or collection.hide_render!=hidden or obj.hide_viewport!=hidden or obj.hide_render!=hidden:raise ValueError('default layer visibility differs')
        if len(obj.data.materials)!=1:raise ValueError('exactly one solid material required')
        material=obj.data.materials[0]
        if material is None or not material.use_nodes:raise ValueError('node material required')
        principled=[node for node in material.node_tree.nodes if node.type=='BSDF_PRINCIPLED']
        if len(principled)!=1:raise ValueError('unique Principled material required')
        diffuse=list(material.diffuse_color);base=list(principled[0].inputs['Base Color'].default_value)
        if any(abs(a-b)>1e-6 for values in (diffuse,base) for a,b in zip(values,color)):raise ValueError('semantic layer material color differs')
        result.append(dict(object=obj.name,title=source['title'],collection=name,hide_render=hidden,hide_viewport=hidden,color=list(color),material=material.name))
    return result


def main():
    mode,base,payload,expected,blend,report,legacy_path,legacy_sha=sys.argv[sys.argv.index('--')+1:]
    if digest(payload)!=expected or digest(legacy_path)!=legacy_sha:raise ValueError('bound worker or payload changed')
    scenes=json.loads(Path(payload).read_text(encoding='utf-8'))['scenes']
    if len(scenes)!=2 or {s['title'] for s in scenes}!=set(STYLES):raise ValueError('exact inferred and observed title pair required')
    spec=importlib.util.spec_from_file_location('_completion_legacy_geometry_check',legacy_path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    if mode=='create':
        bpy.ops.wm.open_mainfile(filepath=base,use_scripts=False)
        before=legacy.check(scenes)
        materials=[]
        for index,source in enumerate(scenes):
            obj=bpy.data.objects.get('PLY_'+str(index+1).zfill(2));name,color,hidden=STYLES[source['title']]
            if bpy.data.collections.get(name) is not None:raise ValueError('presentation collection already exists in base')
            collection=bpy.data.collections.new(name);bpy.context.scene.collection.children.link(collection)
            for old in list(obj.users_collection):old.objects.unlink(obj)
            collection.objects.link(obj)
            collection.hide_viewport=hidden;collection.hide_render=hidden
            obj.hide_viewport=hidden;obj.hide_render=hidden
            obj.color=color
            if len(obj.data.materials)!=1:raise ValueError('expected original single-material export')
            material=obj.data.materials[0]
            if material in materials:raise ValueError('shared material would alias semantic colors')
            materials.append(material);material.diffuse_color=color
            principled=[node for node in material.node_tree.nodes if node.type=='BSDF_PRINCIPLED']
            if len(principled)!=1:raise ValueError('original Principled material unavailable')
            principled[0].inputs['Base Color'].default_value=color
            if not hidden:bpy.context.view_layer.objects.active=obj
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':area.spaces.active.shading.color_type='MATERIAL'
        checks=legacy.check(scenes)
        if checks!=before:raise ValueError('geometry evidence changed while styling')
        presentation=check_style(scenes)
        bpy.ops.wm.save_as_mainfile(filepath=blend,check_existing=True)
    elif mode=='verify':
        bpy.ops.wm.open_mainfile(filepath=blend,use_scripts=False)
        checks=legacy.check(scenes);presentation=check_style(scenes)
    else:raise ValueError('unknown presentation mode')
    if digest(payload)!=expected or digest(legacy_path)!=legacy_sha:raise ValueError('input changed during presentation')
    with Path(report).open('x',encoding='utf-8') as stream:
        json.dump(dict(status='PASS',mode=mode,payload_sha256=expected,scenes=checks,presentation=presentation,blender_version=bpy.app.version_string,geometry_changed=False,metadata_changed=False,appearance_semantics='Solid semantic colors, not recovered photo texture; observed-depth layer hidden by default, inferred completion visible.'),stream,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':main()
