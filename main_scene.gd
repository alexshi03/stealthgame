extends Node3D

const GameOverScreen = preload("res://UI/game_over_screen.tscn")

@onready var enemy = $enemy

func _ready() -> void:
	enemy.connect("player_caught", Callable(self, "_on_player_caught"))
	
func _on_player_caught():
	var game_over = GameOverScreen.instantiate()
	add_child(game_over)
	game_over.set_title(false)
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	get_tree().paused = true
	print("caught!")
	


func _on_area_3d_body_entered(body: Node3D) -> void:
	if body.name == "player":
		var game_over = GameOverScreen.instantiate()
		add_child(game_over)
		game_over.set_title(true	)
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		get_tree().paused = true
		print("won!")
