extends CharacterBody3D

@onready var get_nav_map = get_parent().get_node("NavigationRegion3D")
@onready var nav_agent = get_node("NavigationAgent3D")
@onready var FOV_caster = get_node("FOV_cast")
@onready var capture_viewport = get_node("SubViewport2")

@onready var anim_player = get_node("AnimationPlayer")

@export var speed = 50

var all_points = []
var next_point = 0

var in_pursuit = false
var is_at_post = false
var seen_player = false
@export var is_patrolling_guard = false

#TODO: add last known position

func _ready():
	print(OS.get_user_data_dir())
	add_to_group("enemy")
	
	for x in get_parent().get_node("all_patrolling_points").get_children():
		all_points.append(x.global_position + Vector3(0,1,0))


func _physics_process(delta: float) -> void:
	if Input.is_action_just_pressed("ui_down"):
		in_pursuit = false
	
	if Input.is_action_just_pressed("screenshot"):
		take_screenshot()
		
	if is_patrolling_guard or in_pursuit:
		check_alert_state(delta)
	
	if not is_patrolling_guard and not in_pursuit:
		back_to_post(delta)

	move_and_slide()
	
func check_alert_state(delta):
	if not in_pursuit:
		patrolling(delta)
	else:
		pursuit_state(delta)

func patrolling(delta):
	await get_tree().process_frame
	
	var dir_dir
	
	nav_agent.target_position = all_points[next_point]
	dir_dir = nav_agent.get_next_path_position() - global_position
	
	dir_dir = dir_dir.normalized()
	
	velocity = velocity.lerp(dir_dir * speed * delta, 1.0)
	
	dir_dir.y = 0
	
	look_at(global_transform.origin + dir_dir)
	
	if not anim_player.is_playing() or anim_player.current_animation != "Girl_Anim_Walk":
		anim_player.play("Girl_Anim_Walk")

	
func pursuit_state(delta):
	var enemy_current_map_pos = global_position
	var current_target = nav_agent.get_next_path_position()
	var change_dir = (current_target -enemy_current_map_pos).normalized()
	
	velocity = change_dir * speed * delta
	
	look_at(Vector3(Global.player_current_pos.x,self.global_transform.origin.y, Global.player_current_pos.z), Vector3(0,1,0))
	
	if not anim_player.is_playing() or anim_player.current_animation != "Girl_Anim_Walk":
		anim_player.play("Girl_Anim_Following")


func back_to_post(delta):
	if not is_at_post:
		await get_tree().process_frame
	
		var dir_dir
		
		nav_agent.target_position = all_points[0]
		dir_dir = nav_agent.get_next_path_position() - global_position
		
		dir_dir = dir_dir.normalized()
		
		velocity = velocity.lerp(dir_dir * speed * delta, 1.0)
		
		dir_dir.y = 0
		
		look_at(global_transform.origin + dir_dir)
	
	if is_at_post:
		velocity = Vector3(0, 0, 0)
	


func _on_navigation_agent_3d_target_reached() -> void:
	if is_patrolling_guard:
		next_point += 1
		
		if next_point >= all_points.size():
			next_point = all_points[-1]
			next_point = 0
		
		if not is_patrolling_guard and not in_pursuit:
			is_at_post = true


func _on_timer_timeout() -> void:
	check_sight()
	get_new_target(Global.player_current_pos)
	
func check_sight():
	if seen_player:
		print(Global.player_current_pos)
		FOV_caster.look_at(Global.player_current_pos, Vector3(0,1,0))
		
	if FOV_caster.is_colliding():
		var collider = FOV_caster.get_collider()
		
		if collider.is_in_group("player"):
			
			in_pursuit = true
			is_at_post = false


func _on_enemy_fov_body_entered(body: Node3D) -> void:
	if body.is_in_group("player"):
		print("player seen")
		seen_player = true

func _on_enemy_fov_body_exited(body: Node3D) -> void:
	if body.is_in_group("player"):
		seen_player = false
		in_pursuit = false
		
func get_new_target(new_target):
	nav_agent.set_target_position(new_target)
	
func take_screenshot():
	var img = capture_viewport.get_texture().get_image()
	img.flip_y()
	var file_path = "user://screenshot_" + str(Time.get_ticks_msec()) + ".png"
	img.save_png(file_path)
	print("Screenshot saved at ", file_path)
	print("user is ", OS.get_user_data_dir())
