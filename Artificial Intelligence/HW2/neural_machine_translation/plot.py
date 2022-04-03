import matplotlib.pyplot as plt
import json
import os

scores = []
for i in range(1, 21):
    filename = f'valid_beam_{i}.score'
    with open(os.path.join('outputs', filename), 'r') as f:
        content = json.load(f)
    scores.append(content['score'])

plt.plot(list(range(1, 21)), scores)
plt.title('BLEU Score per Beam Width')
plt.ylabel('Score')
plt.xlabel('Beam Width')
plt.savefig(os.path.join('outputs', 'valid_beam_plot.png'), bbox_inches='tight')
