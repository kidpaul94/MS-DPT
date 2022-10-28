import bpy
import math
import random
import time
from mathutils import Euler, Color
from pathlib import Path

def randomly_rotate_object(obj_to_change): 
    """ Applies a radom rotation to an object """
    random_rot = (random.random() * 2 * math.pi, random.random() * 2 * math.pi, random.random() * 2 * math.pi)
    obj_to_change.rotation_euler = Euler(random_rot, 'XYZ')
    
def randomly_change_color(material_to_change):
    """ Change the Principled BSDF color of a material to a random color """
    color = Color()
    hue = random.random() # We are using alternative color representation here (HSV)
    saturation = 1 #random.random()
    value = 1 #random.random() 
    color.hsv  = (hue, saturation, value)
    rgba = [color.r, color.g, color.b, 1]
    """Without texture image applied to objects"""
    #material_to_change.node_tree.nodes['Principled BSDF'].inputs[0].default_value = rgba
    """With texture image applied to objects"""
    material_to_change.node_tree.nodes["ColorRamp"].color_ramp.elements[0].color = rgba 
    
def randomly_translate_object(obj_to_change): 
    """ Applies a radom translation to an object """
    random_trans = (random.uniform(-4, 4), random.uniform(-5, 5), random.uniform(-2, 8)) # Values can be adjusted if needed
    obj_to_change.location = random_trans



for i in range(100):
    change_dynamic = str(i + 7801)
    bpy.context.scene.node_tree.nodes['File Output'].file_slots[0].path = "Mask_" + change_dynamic # Output path for grayscale images
    randomly_rotate_object(bpy.context.scene.objects['MDP TrainingModel'])
    randomly_rotate_object(bpy.context.scene.objects['MDP TrainingModel.001'])
    randomly_translate_object(bpy.context.scene.objects['MDP TrainingModel'])
    randomly_translate_object(bpy.context.scene.objects['MDP TrainingModel.001'])
    randomly_change_color(bpy.data.materials['Material'])
    string = '/home/kidpaul/Python/training/Image_' + change_dynamic + '.jpg' # Output path for RGB images
    bpy.context.scene.render.filepath = string
    bpy.ops.render.render(write_still=True)

# Use this whenever you need to regenerate single image
"""
change_static = str(7600)
bpy.context.scene.node_tree.nodes['File Output'].file_slots[0].path = "Mask_" + change_static
randomly_rotate_object(bpy.context.scene.objects['MDP TrainingModel'])
randomly_rotate_object(bpy.context.scene.objects['MDP TrainingModel.001'])
randomly_translate_object(bpy.context.scene.objects['MDP TrainingModel'])
randomly_translate_object(bpy.context.scene.objects['MDP TrainingModel.001'])
randomly_change_color(bpy.data.materials['Material'])
string = '/home/kidpaul/Python/training/Image_' + change_static + '.jpg'
bpy.context.scene.render.filepath = string
bpy.ops.render.render(write_still=True)
"""
