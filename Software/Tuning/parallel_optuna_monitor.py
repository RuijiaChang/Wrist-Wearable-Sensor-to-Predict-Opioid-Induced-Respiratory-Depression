import optuna
import time

while True:
    try:
        study = optuna.load_study(
            study_name="weight_CW_F11",
            storage="sqlite:///optuna.db"
        )
        print(f"Trial length: {len(study.trials)}")
    except KeyError as e:
        print(f"Study not found: {e}")
    time.sleep(10)