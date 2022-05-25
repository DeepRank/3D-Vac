import sys 

prefix = sys.argv[1];
file_path = sys.argv[2];
rows = "";

with open(file_path) as file:
    line_count = 0;
    for line in file:
        row = line.split(",");
        if line_count == 0: # add the ID header
            row.insert(0, "ID")
        else:
            row.insert(0, prefix+"_"+str(line_count));
            #row[0] = prefix+"_"+str(line_count)
        line_count+=1
        rows = rows + ",".join(row)
    print("number of lines: ", line_count)
print("last row of the csv file:")
print(rows[-1]);

with open(file_path, "w") as file:
    file.write(rows)
