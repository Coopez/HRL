# HRL
Toy environment (and other stuff?) for Hierarchical RL

The environment consists of a 2D torch tensor of dimension 2xN that contains the coordinates of the grid, and another tensor of dimension N that contains the corresponding gas concentration values. The tensors are contained in the class as env.coordinates and env.values. When creating an environment, the type can be specified to be either elliptical or anistropic. Elliptical is smoother and nicer to look at.

Example usage:
# DEFINE RANDOM ENVIRONMENT: Grid, plume, train coords/values
	env = class_environment.get_random_env(leak_len=40, type='elliptical')

# DEFINE DETERMINISTIC ENVIRONMENT:
	# Grid
	leak_len = 40 # minimum dilution 0.0025
	grid_len = leak_len*math.sqrt(2)
	xmin = -grid_len; xmax = grid_len; ymin = -grid_len; ymax = grid_len
	grid_bounds = [(xmin, xmax), (ymin, ymax)]
	grid_resolution = grid_len*2+1

	# Scenario type
	env_params = {
  		#'type': 'anisotropic'
  		'type': 'elliptical'
	}
	# Current params
	env_params['current_strength'] = 10 # anistropic: [1 20],  elliptic: [1 100]
	env_params['current_direction'] = -120  # [-180 180]

	# Plume params
	env_params['dilution'] = 0.1   # anistropic: [0.0025 0.1], corresponding to leak_lens of [40 5]
                                # elliptic: [0.01 0.1]
	env_params['location_initial'] = [-21, -13]
	env_params['semi_major_axis'] = 20.0
	env_params['semi_minor_axis'] = 5.0

	# Get environment
	env = class_environment.Environment(grid_bounds, grid_resolution, env_params)

 	# Visualize
 	#fig_env = my_env.visualize_3D()
 	#fig_env = my_env.visualize_2D()
	#fig_env.show()
