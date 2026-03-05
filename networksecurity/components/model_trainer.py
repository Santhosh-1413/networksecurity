import os
import sys
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import mlflow
import dagshub

dagshub.init(repo_owner='Santhosh-1413', repo_name='networksecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        from sklearn.model_selection import GridSearchCV
        params = {
            "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
            "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
            "Gradient Boosting": {'learning_rate': [0.1, 0.01, 0.05, 0.001], 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9], 'n_estimators': [8, 16, 32, 64, 128, 256]},
            "Logistic Regression": {},
            "AdaBoost": {'learning_rate': [0.1, 0.01, 0.5, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
        }

        from networksecurity.utils.main_utils.utils import load_numpy_array_data
        model_report = {}
        for model_name, model in models.items():
            param = params[model_name]
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(model, test_metric)
            model_report[model_name] = test_metric.f1_score

        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
        save_object("final_model/model.pkl", obj=best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1],
            )
            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)