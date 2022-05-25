from os import walk,remove
import datetime 
import json

json_path = "/projects/0/einf2380/data/modelling_logs/"
json_files = [];
json_objects = [];
outputs = {
    "db_time": [],
    "per_node_time": [],
    "creating_wrapper_targets": [],
    "modeling_time": [],
    "number_of_cases": []
};
for path, dir, filenames in walk(json_path): ## get the list of json files
    for filename in filenames: 
        if "wrapper_output" in filename:
            json_files.append(filename);

for json_file in json_files: ##load the json content of each files in json_files
    with open(json_path+json_file) as f:
        json_str = f.read();
        json_objects.append(json.loads(json_str))

for obj in json_objects: ## populate the outputs with the values in each files
    for k,v in obj.items():
        outputs[k].append(v)

outputs = {k:sum(v)/len(outputs) for k,v in outputs.items()} ##make the mean value of each output

dt = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M");
with open(json_path+f"dispatched_jobs_logs_from_{dt}.json", "w") as f: ## save the output in a log file
    f.write(json.dumps(outputs));

for f in json_files: ##remove the json files
    remove(json_path + f);

print(f"number of jobs: {len(json_files)}");
for k,v in outputs.items():
    print(f"the {k} has a mean value of {v}")
