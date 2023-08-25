import matplotlib.pyplot as plt

# Filtra os dados para o cluster desejado (por exemplo, cluster_label = 1)
cluster_label = 0
cluster_data = nutrient_counts_by_cluster[nutrient_counts_by_cluster['cluster_label'] == cluster_label]

plt.figure(figsize=(10, 6))

plt.bar(cluster_data['nutrient_name'], cluster_data['count'])

plt.xlabel('Nutrientes')
plt.ylabel('Quantidade')
plt.title(f'Nutrientes mais Frequentes no Cluster {cluster_label}')
plt.xticks(rotation=45)

# Exibe o gráfico
plt.tight_layout()
plt.show()