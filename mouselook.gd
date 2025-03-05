extends Node

@export var player: CharacterBody3D
@export var camera_pivot: Node3D 

var mouse_sensitivity = 0.002
var pitch = 0.0

func _ready():
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _input(event):
	if event is InputEventMouseMotion:
		player.rotate_y(-event.relative.x * mouse_sensitivity)  
		pitch -= event.relative.y * mouse_sensitivity 
		pitch = clamp(pitch, deg_to_rad(-80), deg_to_rad(80)) 
		camera_pivot.rotation.x = pitch

	if Input.is_action_just_pressed("u	i_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE 
