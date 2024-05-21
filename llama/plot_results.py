import matplotlib.pyplot as plt
import json
import sys
import numpy as np

# Add the paths of your JSON files in this list
json_files = sys.argv[1:]

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f) 

        update_steps = np.linspace(data["update_step_begin"], data["update_step_end"], len(data["update_loss"]))
        update_loss = data["update_loss"]

        if len(json_file.split("/"))>=3:
            label=json_file.split("/")[-3]
        elif len(json_file.split("/"))>=2:
            label=json_file.split("/")[-2]
        else:
            label=json_file
        # Plotting data
        plt.plot(update_steps, update_loss, label=label)

        if "test_loss" in data:
            test_loss_pair = data["test_loss"]
            test_steps = [pair[0] for pair in test_loss_pair ]
            test_loss = [pair[1] for pair in test_loss_pair ]
            plt.plot(test_steps, test_loss, "o", label=label)

# Add title and labels      
plt.title('Update Loss Over Update Steps') 
plt.xlabel('Update Steps') 
plt.ylabel('Update Loss')  

# Add legend
plt.legend()

# Show the plot
plt.show()
