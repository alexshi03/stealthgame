extends CanvasLayer

@onready var title = $PanelContainer/MarginContainer/Rows/Title

func _on_restart_pressed() -> void:
	get_tree().paused = false
	get_tree().change_scene_to_file("res://main_scene.tscn")


func _on_quit_pressed() -> void:
	get_tree().quit()

func set_title(win: bool):
	if win:
		title.text = "YOU WIN!"
		title.modulate = Color.GREEN
	else:
		title.text = "YOU LOSE!"
		title.modulate = Color.RED
