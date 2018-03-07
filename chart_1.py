import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')

df = pd.read_csv('cart_pole_v4_v8.txt', sep = ';', names = ["a", "b", "c", "d"])

print df["a"][0]
print df["b"][0]
print df["c"][0]

generation = []
best_individual_score = []
second_best_individual = []
third_best_individual = []
average_generation_score = []
for index, row in df.iterrows():
	values = row["b"].split('[')[1].split(']')[0]
	values = values.split(',')
	values = [float(element.strip()) for element in values]
	best_individual_score.append(values[0])
	second_best_individual.append(values[1])
	third_best_individual.append(values[2])
	average_generation_score.append(float(row["c"]))

# print average_generation_score

plot_df = pd.DataFrame({
	"Best Individual in Generation": best_individual_score[:10],
	"Second Individual in Generation": second_best_individual[:10],
	"Third Individual in Generation": third_best_individual[:10],
	"Average Generation Score": average_generation_score[:10],
	})

print plot_df
plot_df.plot(figsize = (8, 6), title = "Performance (frames) per Generation - Agent in the Cartpole Environment", grid = True)

plt.show(block=True)

# plt.figure(figsize=(10,8))
# plt.xlabel('Features')
# plt.ylabel('Chi-Square')
# plt.title('Sorted - Chi-Square per feature')

# sns_plot=sns.barplot(ordered_feature_labels, ordered_chi_result)
# sns_plot.set_xticklabels(feature_labels, rotation=90, fontsize=7)
# sns_plot.figure.savefig("chi_square.png")










