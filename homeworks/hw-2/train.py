import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    train_data_path = os.path.join(data_path, "train.pkl")
    val_data_path = os.path.join(data_path, "val.pkl")

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.set_tag("developer", "sergei")

        mlflow.log_param("train-data-path", train_data_path)
        mlflow.log_param("valid-data-path", val_data_path)

        X_train, y_train = load_pickle(train_data_path)
        X_val, y_val = load_pickle(val_data_path)

        params = {
            "max_depth": 10,
            "random_state": 0
        }
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
