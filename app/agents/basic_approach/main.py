import matplotlib.pyplot as plt
from data.text_info import text_profile
from visualization.freq_visual import create_hist


dct = text_profile("C:\\Users\\Kirushka\\Desktop\\1.py")
fix, axes = plt.subplots(3, 1, figsize=(20, 10))
char_commons = dct["char_cnts"].most_common(15)
bigram_commons = dct["bigram_cnts"].most_common(15)
trigram_commons = dct["trigram_cnts"].most_common(15)

create_hist(axes[0],
            [item[0] for item in char_commons],
            [item[1] for item in char_commons],
            title="char_cn",
            color="orange")
create_hist(axes[1],
            [item[0] for item in bigram_commons],
            [item[1] for item in bigram_commons],
            title="bigram_cn",
            color="blue")
create_hist(axes[2],
            [item[0] for item in trigram_commons],
            [item[1] for item in trigram_commons],
            title="trigram_cn",
            color="red")

plt.show()