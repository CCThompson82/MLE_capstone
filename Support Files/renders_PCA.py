def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (20,10))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar', legend=False);
	ax.set_ylabel("Feature Weights", fontsize=16)
	ax.set_ylim(-.5,0.6)
	ax.set_xticklabels(dimensions, rotation=0, fontsize=16)
	patches, labels = ax.get_legend_handles_labels()
	ax.legend(patches, labels, loc= 'lower center', ncol=10)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-.1, ax.get_ylim()[1] + 0.01, "Explained Variance\n %.4f"%(ev), fontsize=16)

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)
