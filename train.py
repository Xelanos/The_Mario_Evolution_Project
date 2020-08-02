import argparse
from pandas import DataFrame
import gym_super_mario_bros
import human_playing


DEFAULT_ENVIRONMENT = 'SuperMarioBros-v0'
DEFAULT_STEP_LIMIT = 2000
DEFAULT_NO_ADVANCE_STEP_LIMIT = 100
AGENTS = ['human', 'genetic']
TRIALS = 10

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train agents.")
    parser.add_argument("-agent", dest='agent', choices=AGENTS, default=AGENTS[0],
                        help="Chose kind of agent for training.")
    parser.add_argument("-n", dest='num_of_trials', type=int, default=TRIALS, help="Number of trails.")
    parser.add_argument("-s", '-time_scale', "-steps_limit", dest='steps_limit', default=DEFAULT_STEP_LIMIT, type=int,
                        help="The maximal frames to a trail.")
    parser.add_argument("-no_action_limit", "-no_advance_limit", "-standing_limit", "-no_progress_limit",
                        dest='standing_steps_limit', default=DEFAULT_NO_ADVANCE_STEP_LIMIT, help="Limit the number of"
                        " steps the agent allow not to change the x position.")
    parser.add_argument("-d","-allow_death", dest="allow_death", action='store_false',
                        help="Allow agent to die in a trail.")
    parser.add_argument("-e", "-env", dest="env", default=DEFAULT_ENVIRONMENT, help="The environment ID to play")

    args = parser.parse_args()
    if args.steps_limit < 1:
        parser.error("steps limit must to be positive.")

    return args


if __name__ == "__main__":
    output_data = DataFrame(columns=['avg_reward', 'steps', 'score', 'deaths', 'finish_level'])
    args = parse_arguments()
    if args.agent == "human":
        for trial in range(args.num_of_trails, args.steps_limit, args.standing_steps_limit):
            outcome = human_playing.run(gym_super_mario_bros.make(args.env))
            output_data[trial] = outcome

