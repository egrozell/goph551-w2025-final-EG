import final_project.model_from_bin as load_model

dir = "./models/Marmousi/"
load_model.get_dims(dir)
load_model.read_bin(dir)
