extends CharacterBody3D

@onready var get_nav_map = get_parent().get_node("NavigationRegion3D")
@onready var nav_agent = get_node("NavigationAgent3D")
@onready var FOV_caster = get_node("FOV_cast")
@onready var capture_viewport = get_node("SubViewport2")

@onready var anim_player = get_node("AnimationPlayer")
@onready var cam = get_node("SubViewport2/Camera3D")

@export var speed = 50

var all_points = []
var next_point = 0

var in_pursuit = false
var is_at_post = false
var seen_player = false
@export var is_patrolling_guard = false

signal player_caught

var current_pos = 0
var model_size = Vector2(512, 512)
var screen_size = 0


#TODO: add last known position

func _ready():
	print(OS.get_user_data_dir())
	add_to_group("enemy")
	
	for x in get_parent().get_node("all_patrolling_points").get_children():
		all_points.append(x.global_position + Vector3(0,1,0))
		
	screen_size = get_viewport().get_visible_rect().size


func _physics_process(delta: float) -> void:
	#if Input.is_action_just_pressed("ui_down"):
		#in_pursuit = false
	
	if Input.is_action_just_pressed("screenshot"):
		#take_screenshot()
		send_to_server()
		
	if is_patrolling_guard or in_pursuit:
		check_alert_state(delta)
	
	if not is_patrolling_guard and not in_pursuit:
		back_to_post(delta)
	
	var collision = move_and_collide(velocity * delta)
	if collision and collision.get_collider().name == "player":
		emit_signal(("player_caught"))

	move_and_slide()
	
func check_alert_state(delta):
	if not in_pursuit:
		patrolling(delta)
	else:
		pursuit_state(delta)

func patrolling(delta):
	await get_tree().process_frame
	
	$Timer.wait_time = 1
	
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
	$Timer.wait_time = 0.5
	nav_agent.set_target_position(current_pos)
	var enemy_current_map_pos = global_position
	var current_target = nav_agent.get_next_path_position()
	var change_dir = (current_target -enemy_current_map_pos).normalized()
	
	velocity = change_dir * speed * delta
	
	look_at(Vector3(current_pos.x,self.global_transform.origin.y, current_pos.z), Vector3(0,1,0))
	
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
	send_to_server()
	#get_new_target(Global.player_current_pos)
	
func check_sight(parsed):
	if not parsed:
		return 
		
	if parsed["confidence"] > 0:
		seen_player = true
		print("seen!")
	else:
		seen_player = false
		in_pursuit = false
	
	if seen_player:
		var box = parsed["boxes"]
		var center = get_box_center(box)
		var world_pos = screen_to_world(cam, center)
		
		print(Global.player_current_pos)
		print(world_pos)
		current_pos = world_pos
		FOV_caster.look_at(world_pos, Vector3(0,1,0))
		in_pursuit = true
		is_at_post = false
		
func get_new_target(new_target):
	nav_agent.set_target_position(new_target)
	
func take_screenshot():
	var img = capture_viewport.get_texture().get_image()
	img.flip_y()
	var file_path = "user://screenshot_" + str(Time.get_ticks_msec()) + ".png"
	img.save_png(file_path)
	print("Screenshot saved at ", file_path)
	print("user is ", OS.get_user_data_dir())
	
func send_to_server():
	var img = capture_viewport.get_texture().get_image()
	img.flip_y()
	var png_bytes = img.save_png_to_buffer()

	var headers = [
		"Content-Type: image/png"
	]

	$HTTPRequest.request_raw(
		"http://127.0.0.1:5000/predict",
		headers,
		HTTPClient.METHOD_POST,
		png_bytes
	)


func _on_http_request_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	if response_code != 200:
		print("Request failed with code:", response_code)
		return
		
	var json_string = body.get_string_from_utf8()
	var parsed = JSON.parse_string(json_string)
	
	if parsed == null:
		print("Failed to parse JSON")
		return
	
	print(parsed)
	
	check_sight(parsed)
	return parsed
	
func get_box_center(box):
	var center = Vector2((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
	
	center.y = model_size.y - center.y 
	
	return center

func screen_to_world(camera: Camera3D, screen_pos: Vector2) -> Vector3:
	var ray_origin = camera.project_ray_origin(screen_pos)
	var ray_dir = camera.project_ray_normal(screen_pos)
	var ground_y = 1
	var distance = (ground_y - ray_origin.y) / ray_dir.y
	return ray_origin + ray_dir * distance

	
