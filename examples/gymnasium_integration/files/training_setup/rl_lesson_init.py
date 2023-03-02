
import random


def rl_lesson_init(rl_lesson_config):
    """ Startup a set of sim config values given the 'rl_lesson_config' values OR ranges. """

    reset_config = {}
    
    for k,v_d in rl_lesson_config.items():

        # choose a random value from the list of values and add to config
        if v_d.get('values', None) is not None:
            value = random.choice(v_d["values"])
            # add randomize value to episode config
            reset_config[k] = value
            continue
        
        # choose a random value from the range
        v_min = v_d.get('min', None)
        v_max = v_d.get('max', None)
        if v_min is not None and v_max is not None:
            value = random.uniform(v_min, v_max)
            # add randomize value to episode config
            reset_config[k] = value
            continue
        
        print(f"Unknown config key ({k}) with values: {v_d}")
    
    return reset_config