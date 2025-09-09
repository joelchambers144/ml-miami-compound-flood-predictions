import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go

def plot_model_predictions(labels, predictions, title, x_label, y_label, plot_file_name, legend_location = 'best'):
  """Plots model predictions and compares to testing labels to evaluate the model performance

	Parameters:
        labels (list): list of labels

        predictions (list): list of model predictions

        title (string): plot title

        x_label (string): x-axis label

        y_label (string): y-axis label

        plot_file_name (string): file name for plot. Must have appropriate image extension (eg. 'png', 'pdf', 'svg', ...)

        legend_location (string): 'best' (Axes only), 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                  'center left', 'center right', 'lower center', 'upper center', 'center'. Defaults to 'best'
	"""
  fig, ax = plt.subplots(1, figsize=(25, 10))

  # Plot the observations and model predictions on the same plot
  plt.plot(labels, label='Observed')
  plt.plot(predictions, label='Predicted')

  plt.title(title, fontsize=30)

  ax.set_xlabel(x_label, fontsize=22)
  ax.set_ylabel(y_label, fontsize=22)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.legend(fontsize = 20, loc = legend_location)

  plt.savefig(plot_file_name, bbox_inches='tight')

  plt.show()

  plt.close()


def plot_model_loss(model_history, plot_file_name, legend_location = 'best'):
  """Plots model training loss and validation loss on the same plot

	Parameters:
        model_history (keras.src.callbacks.History): model history

        plot_file_name (string): file name for plot. Must have appropriate image extension (eg. 'png', 'pdf', 'svg', ...)

        legend_location (string): 'best' (Axes only), 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                  'center left', 'center right', 'lower center', 'upper center', 'center'. Defaults to 'best'
	"""
  fig, ax = plt.subplots(1, figsize=(25, 10))

  # Plot training & validation loss values
  plt.plot(model_history.history['loss'])
  plt.plot(model_history.history['val_loss'])

  lowest_val_loss = min(model_history.history['val_loss'])
  best_epoch = model_history.history['val_loss'].index(lowest_val_loss)

  ax.axvline(x = best_epoch, color='r', linestyle='--')

  plt.title('Model loss', fontsize=30)

  ax.set_xlabel('Epoch', fontsize=22)
  ax.set_ylabel('Loss', fontsize=22)

  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)

  plt.legend(['Train', 'Validation', 'Lowest Validation Loss'], fontsize = 20, loc = legend_location)

  plt.savefig(plot_file_name, bbox_inches='tight')

  plt.show()

  plt.close()


def plot_interactive_time_series(dfs, title, x_label, y_label, plot_file_name, mode = 'lines', show = True):
  """Plots list of pandas dataframes datetime data on the same interactive Plotly chart.
  Saves the resulting plot in the user-specified path.

	Parameters:
        dfs (list): list of pandas dataframes. Must contain columns: 'data', 'dash_style', 'line_color', and 'legend_label'

        title (string): plot title

        x_label (string): x-axis label

        y_label (string): y-axis label

        plot_file_name (string): file name for plot. Must have appropriate image extension ('.html')

        mode (string): Choose between 'markers', 'lines', and 'lines+markers'. Defaults to 'lines'

        show (bool): Whether or not to display the plot on the screen. If the plot contains a large amount of data it
                     is recommended to set show = False. The code may crash if you do not
	"""
  fig = go.Figure()

  for df in dfs:
    fig.add_trace(go.Scatter(x = df.index, y = df['data'], mode = mode,
                             line = dict(dash=df['dash_style'][0], color=df['line_color'][0], width=3.5),
                             name = df['legend_label'][0]))

  fig.update_layout(
      title=go.layout.Title(
                    text = title,
                    font=dict(
                        family = 'Arial',
                        size = 25,
                        color = 'black'
                    ),
                    xref = 'paper',
                    x = 0.5
      ),

      xaxis=dict(title = x_label,
                 title_font = dict(
                        family = 'Arial',
                        size = 22,
                        color = 'black'
                    )
                 ),
      yaxis=dict(title = y_label,
                 title_font = dict(
                        family = 'Arial',
                        size = 22,
                        color = 'black'
                    )
                 ),
      # Update tickfonts
      xaxis_tickfont=dict(size=18),
      yaxis_tickfont=dict(size=18),
      legend=dict(
          font=dict(
              family='Arial',
              size=18,
              color='black'
          )
      ),

      hovermode='x unified'
  )

  fig.update_traces(connectgaps=False)

  fig.write_html(plot_file_name, include_plotlyjs='cdn')

  if show is True:
    fig.show()


def plot_preds_vs_obs_interactive(predictions, labels, datetime_index, experiment, lead_time):
  dfs_to_plot = []

  # Add lead time to reference times to get verified times
  datetime_index = datetime_index + pd.Timedelta(hours=lead_time)

  # Predictions
  csv_file_name = f'{experiment}/{lead_time}hr_{experiment}_predictions.csv'
  predictions_df = pd.DataFrame({'data': predictions.flatten()}, index=datetime_index)
  predictions_df.to_csv(csv_file_name)
  predictions_df['dash_style'] = 'dash'
  predictions_df['legend_label'] = 'Predicted'
  predictions_df['line_color'] = 'orange'
  dfs_to_plot.append(predictions_df)


  # Observed
  observed_df = pd.DataFrame({'data': labels.flatten()}, index = datetime_index)
  observed_df['dash_style'] = 'solid'
  observed_df['legend_label'] = 'Observed'
  observed_df['line_color'] = 'blue'
  dfs_to_plot.append(observed_df)


  # Plot
  title = f'{lead_time}hr Groundwater Level MLP Model Observed vs Predicted'
  x_label = 'Datetime (UTC)'
  y_label = 'Elevation in meters (NAVD88)'
  plot_file_name = f'{experiment}/{lead_time}hr_{experiment}_obsvspred.html'
  plot_interactive_time_series(dfs_to_plot, title, x_label, y_label, plot_file_name, show = True)