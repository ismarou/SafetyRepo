import subprocess
import os
import sys
import numpy as np

"""
Run one after another in tmux on ilabs
"""

def submit_slurm_job(script):
    #subprocess.run(['srun', '-G', '1', '--pty']+script)
    subprocess.run('srun '+'-G '+'1 '+'--pty '+script, shell=True)

#So first it is 0 so it uses the first list nums and sets the first current point value
#Then it goes to 1 if there is more to go to and there is uses the second variable's list nums and sets the current point of that variable to that value
def recurse(commands, base, name, list_nums, current_point, current_depth, desired_depth):
    for i in list_nums[current_depth]:
      if current_depth < desired_depth:
        current_point[current_depth] = i
        recurse(commands, base+" --"+name[current_depth]+"="+str(i), name, list_nums, current_point, current_depth+1, desired_depth)
      else:
        commands.append(base+" --"+name[current_depth]+"="+str(i))
        #python train_model.py --config_path=some_config.yaml --exp_name=my_first_exp
      

def main():
    algo = str(sys.argv[1])
    commands=[]
    base=" "
    if algo == "BC":
      base = "python using_planner/BC.py --config_path=./using_planner/cfg/BC_config.yaml"
    if algo == "TD3-BC":
      base = "python using_planner/ORL.py --config_path=./using_planner/cfg/TD3-BC_config.yaml"
      
    if len(sys.argv) > 2:
      name=[]
      list_nums=[]
      i=2
      for _ in range(int(len(sys.argv)/4)):
        #the first value is the name of what we are scanning
        #the second value is the start
        #the third value is the shift per
        #the fourth value is the end
        name.append(str(sys.argv[i]))
        i+=1
        start=float(sys.argv[i])
        i+=1
        num=int(sys.argv[i])
        i+=1
        end=float(sys.argv[i])
        i+=1
        list_nums.append(np.linspace(start, end, num))
        
      current_point=[0] * len(name)
      recurse(commands, base, name, list_nums, current_point, 0, len(name)-1)
    else:
      commands.append(base)
    
    for script in commands:
        print(f"These are the Slurm jobs: {script}")

    for script in commands:
        print(f"Submitting Slurm job: {script}")
        submit_slurm_job(script)
        print(f"Waiting for job to finish: {script}")
        subprocess.run(['srun', '--wait=0', 'sleep', '1'])  # Adjust sleep duration as needed

if __name__ == "__main__":
    main()