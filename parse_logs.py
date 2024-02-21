import re
import pandas as pd

def parse_data(text):
    data = text.strip().split("**************************************************")
    results = []

    for datum in data:
        result = {}
        fields = datum.split("\n")
        for field in fields:
            if field:
                match = re.search(r'^(.*):\s(.*)$', field)
                if match:
                    key = match.group(1)
                    value = match.group(2)
                    try:
                        value = float(value)
                    except ValueError:
                        if value == "True":
                            value = True
                        elif value == "False":
                            value = False
                    result[key] = value
        results.append(result)

    return results

dir = "outputs_early_stop/"
# Read the text file first
with open(dir + "log.txt", "r") as file:
    text = file.read()
    results = parse_data(text)
    print(results)
df = pd.DataFrame(results)

# Save dataframe to csv
df.to_csv(dir + 'results.csv', index=False)