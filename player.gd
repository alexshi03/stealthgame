extends CharacterBody3D

@onready var standing_col = get_node("standing_col")
@onready var crouching_col = get_node("crouching_col")
@onready var cam_pivot = get_node("cam_pivot")

var speed = 3
var jump_vel = 4.5
var mouse_sensitivity = 0.002

var is_crouching = false

func _ready():
	add_to_group("player")
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _physics_process(delta: float) -> void:
	Global.player_current_pos = global_position
	
	control_loop(delta)
	
func _input(event):
	if Input.is_action_just_pressed("toggle_camera"):
		_swap_camera()
	
	if event is InputEventMouseMotion:
		rotate_y(-event.relative.x * mouse_sensitivity)
		cam_pivot.rotate_x(-event.relative.y * mouse_sensitivity)
		#keeps camera from being able to 360, sprite should be parented to camera
		cam_pivot.rotation.x = clamp(cam_pivot.rotation.x,-PI/2, PI/2)
	
	if event.is_action_pressed("crouch"):
		if is_crouching:
			cam_pivot.position.y += 0.5
		else:
			cam_pivot.position.y -= 0.5
		check_crouch_state()
		is_crouching = not is_crouching
	
@onready var player_camera = get_node("cam_pivot/Camera3D")
@export var debug_camera: Camera3D
var using_debug_camera = false
func _swap_camera():
	using_debug_camera = !using_debug_camera
	player_camera.current = !using_debug_camera
	debug_camera.current = using_debug_camera
	
func control_loop(delta):
	if Input.is_action_just_pressed("ui_cancel"):
		#places mouse cursor to center of screen, then locks it to center
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if Input.mouse_mode == Input.MOUSE_MODE_VISIBLE else Input.MOUSE_MODE_VISIBLE
	
	# Add the gravity.
	if not is_on_floor():
		velocity += get_gravity() * delta

	# Handle jump.
	if Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = jump_vel

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	var input_dir := Input.get_vector("move_left", "move_right", "move_forward", "move_backward")
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		velocity.x = direction.x * speed
		velocity.z = direction.z * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed)
		velocity.z = move_toward(velocity.z, 0, speed)

	move_and_slide()
	
func check_crouch_state():
	standing_col.disabled = not is_crouching
	crouching_col.disabled = is_crouching
