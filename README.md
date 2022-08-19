# mujoco_rl_manipulate_unknown_objects
MuJoCo/OpenAI Gym environment for robotic control using Reinforcement Learning. The task is learn to manipulate large unknown objects with underactuated robots.

## Installation 

**Note:** If using GPU, change the torch package reference in [setup.py](setup.py). For more details, see [here](https://pytorch.org/get-started/locally/).

Use pip to install the dependencies.

```
pip install -e .
```

If using GPU you can check if the installation was successful with:
```
python -c "import torch; print(torch.cuda.is_available())"
```

## Run Models

To run the models, *'models/eval_agent.py'* takes the following arguments:

* --train_env - trained model to evaluate, eg "acorn"
* --sim_env - simulated environment, eg "/xmls/acorn_env.xml"
* --render - render environment for visualisation
* --plot_trajectory - plot object and robot trajectories
* --direction - target vector direction, eg 0 or 45

```
python models/eval_agent.py --train_env acorn --sim_env /xmls/acorn_env.xml --render --plot_trajectory --direction 0
```

## Train models

To train the models, *'models/train_agent.py'* takes the following arguments:

* --sim_env - simulated environment, eg "/xmls/acorn_env.xml"
* --task - adds training task name to saved model folder, eg "/reward"
* --name - name of the experiment, eg "sand_ball"
* --direction - target vector direction, eg 0 or 45

```
python models/train_agent.py --sim_env /xmls/bread_crumb_env.xml --task reward --name bread_crumb --direction 0
```

## Results

Acorn environment with forward movement.

![Alt Text](visuals/gifs/acorn_on_acorn_env.gif)

Bread crumb environment with forward movement.

![Alt Text](visuals/gifs/bread_crumb_on_bread_crumb_env.gif)

Sand ball environment with forward movement.

![Alt Text](visuals/gifs/sand_ball_on_sand_ball_env.gif)

Sugar cube environment with forward movement.

![Alt Text](visuals/gifs/sugar_cube_on_sugar_cube_env.gif)

## License

This project is licensed under the MIT License, see the [LICENSE.md](LICENSE.md) file for details.
