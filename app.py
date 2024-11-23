from flask import Flask, render_template, request, redirect, url_for
import os
from Main import load_and_prepare_data, train_model, save_predictions, plot_predictions

app = Flask(__name__)

# Create folders for static resources if they don't exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)  # Ensure the uploads folder exists

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the file from the form
        file = request.files["inputFile"]
        if file:
            # Save the file to a temporary location
            file_path = os.path.join('static', 'uploads', file.filename)
            file.save(file_path)

            # Load, prepare, and train the model using main.py functions
            df = load_and_prepare_data(file_path)
            model, results_df, metrics, future_sales = train_model(df)

            # Format the 'Date' column to a string (YYYY-MM-DD)
            results_df['Date'] = results_df['Date'].dt.strftime('%Y-%m-%d')

            # Round off metrics to 3 decimal places
            metrics['mse'] = round(metrics['mse'], 3)
            metrics['mae'] = round(metrics['mae'], 3)
            metrics['r2'] = round(metrics['r2'], 3)

            # Save predictions to CSV
            predictions_csv_path = os.path.join('static', 'SalesPredictions.csv')
            save_predictions(results_df, predictions_csv_path)

            # Plot predictions and save as an image
            graph_path = os.path.join('static', 'images', 'sales_forecast.png')
            plot_predictions(results_df, graph_path)

            # Redirect to the results page with graph, data, and future sales
            return render_template(
                'results.html',
                graph_path=graph_path,
                metrics=metrics,
                table_data=results_df.to_dict(orient='records'),
                future_sales=future_sales.to_dict(orient='records')  # Pass the future sales data
            )

    return render_template("index.html")

@app.route("/download")
def download():
    # Provide a download link to the generated file
    return '''
        <h3>File Processed Successfully!</h3>
        <a href="/static/SalesPredictions.csv" download>Click here to download the predictions CSV file.</a>
    '''

if __name__ == "__main__":
    app.run(debug=True)
