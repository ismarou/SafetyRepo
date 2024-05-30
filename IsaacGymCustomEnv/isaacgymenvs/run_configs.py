import subprocess
import os
import sys

"""
Want to be able to run jobs as a chosen list of variations
Edit the file to have different presets
Append these different useful options as more parts to the list or write custom option
Then have algorithm sets to run such as training then testing or PPO then SAC, these are just lists of lists
"""

#attr_dict = {'pick':['task=IndustRealTaskFarPegsPick'],
#             'seed_tests':['task.randomize.plug_pos_xy_initial_noise_seed=3'],
#             'test':['test=True']
#            }

def submit_slurm_job(script):
    subprocess.run(['srun', '-G', '1', '--pty']+script)
    #subprocess.run(['squeue', '--user', os.environ['USER'], '--start', '-o', '%i', '--noheader'])

#Attributes applied to each command

def process_attributes():
    start=['python', 'train.py']
    for i in range(1, len(sys.argv)):
      attr = str(sys.argv[i])
      if attr == '--task':
        start.append('task='+str(sys.argv[i+1]))
    start.append('headless=True')
    start.append('wandb_activate=True')
    start.append('wandb_entity=joehdoerr')
    start.append('wandb_project=IndustRealPickCenterDense')
    return start
      

#Algos, the format is returning a [[]] list where each inner list is a command that is in running order
#In each algo, search yourself for the --vartype declaration

def Base():
    command = process_attributes()
    return [command]
    
def pick_PPO_SAC():
    ret=[]
    command = process_attributes()
    command.append(['train=IndustRealTaskFarPegsPickSAC'])
    ret.append(command)
    command = process_attributes()
    command.append(['train=IndustRealTaskFarPegsPickPPO'])
    ret.append(command)
    return ret
    
def test_seeds():
    checkpoint=" "
    for i in range(1, sys.argc):
      attr = str(sys.argv[i])
      if attr == '--checkpoint':
        checkpoint=str(sys.argv[i+1])
    command = process_attributes()
    #need some way that isn't max epochs to run this then save the data in the file
    return command
    
def test_grasp_rew_scale():
    ret=[]
    scale=[0.05, 0.2, 0.5]
    for s in scale:
      command = process_attributes()
      command.append('train.params.config.max_epochs=3000')
      command.append('task.rl.success_bonus='+str(s))
      #command.append( #Since the run outputs come with their .yaml file that ran them, can just add some variable to the yaml to see which it was
      ret.append(command)
    return ret

def main():
    algo = str(sys.argv[1])
    commands=[]
    if algo == "Base":
      commands=Base()
    if algo == "pick_PPO_SAC":
      commands=pick_PPO_SAC()
    
    #python run_configs.py test_grasp_rew_scale --task IndustRealTaskFarPegsPick
    if algo == "test_grasp_rew_scale":
      commands=test_grasp_rew_scale()

    for script in commands:
        print(f"Submitting Slurm job: {script}")
        submit_slurm_job(script)
        print(f"Waiting for job to finish: {script}")
        subprocess.run(['srun', '--wait=0', 'sleep', '1'])  # Adjust sleep duration as needed

if __name__ == "__main__":
    main()