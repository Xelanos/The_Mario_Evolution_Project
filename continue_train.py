import os
import json
import train
from pandas import DataFrame
from mario_evolution import GeneticMario

INPUT_DIR = "train_output_genetic_11-08_17-30"
INPUT_DIR_LAST_GEN = "train_output_genetic_11-08_17-30\gen_25"
OUTPUT_DIR = "train_output_genetic_11-08_17-30"
RENDER_F = 0


def write_summary(args_dict, output_data_frame: DataFrame):
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), 'w') as summary_file:
        summary_file.write("Summary for training genetic agent on {env}".format(env=args_dict['env']))
        summary_file.write(".\n")
        summary_file.write("Ran for {num_of_loops} generations with at most {steps_limit} steps per game.\n".format(
            num_of_loops=args_dict['num_of_loops'], steps_limit=args_dict["steps_limit"]))
        summary_file.write("Limit on no changing the x position is: {standing_limit}\n"
                           "{allow_dying} allow player to die.\n".format(standing_limit=args_dict["standing_steps_limit"],
                                                                allow_dying="Didn't" if args_dict["allow_death"] else "Did"))
        summary_file.write("Initial population is: {i_p}\n"
                               "Elite size is: {e_s}\n"
                               "Random pick size is {r_p}".format(i_p=args_dict["initial_population"], e_s=args_dict["elite_size"],
                                                                  r_p=args_dict["pick_size"]))
        if any(output_data_frame['finish_level']):
            summary_file.write("Agent successfully win the level in some games.\n")
        else:
            summary_file.write("Agent failed to win any games.\n")
        last_gen_outcome = output_data_frame.loc[output_data_frame['generation'] == max(output_data_frame['generation'])]
        summary_file.write("Average performance score of last generation elite is {avg_score}\n".
                           format(avg_score=last_gen_outcome.mean(axis=0)['performance_score']))
        best_result_index = last_gen_outcome['performance_score'].idxmax()
        info = output_data_frame.iloc[best_result_index]
        summary_file.write("Best performance: {best_index} with performance score of"
                           " {performance_score}. ".format(best_index=info['index'],
                                                           performance_score=info['performance_score']))
        if info['finish_level']:
            summary_file.write("Finish level with size {size}. ".format(size=info['finish_status']))
        summary_file.write("Finish in {steps} steps and {deaths} deaths. Score {score} and {coins} coins.\n".format(
            steps=info['steps'], deaths=info['deaths'], score=info['score'], coins=info['coins']))

if __name__ == "__main__":
    with open(os.path.join(INPUT_DIR, "train_arguments.json"), "r") as arguments_file:
        args_dict = json.load(arguments_file)
    model = GeneticMario(mario_environment=args_dict['env'],
                         actions=train.ACTION_SET[args_dict['action_set']],
                         generations=args_dict['num_of_loops'],
                         initial_pop=args_dict["initial_population"],
                         elite_size=args_dict["elite_size"],
                         pick_size=args_dict["pick_size"],
                         steps_scale=args_dict["steps_limit"],
                         allow_death=args_dict["allow_death"],
                         standing_steps_limit=args_dict["standing_steps_limit"],
                         output_dir=OUTPUT_DIR)
    outcomes = model.continue_run(INPUT_DIR_LAST_GEN, render_every=RENDER_F, record_every=0)
    write_summary(args_dict, DataFrame(outcomes))